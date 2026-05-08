import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
from config import groq_request_context
from data_fetch import (
    analyze_user_provided_news,
    extract_intent,
    fetch_realtime_stock_data,
    fetch_stock_or_macro_sentiment,
    find_stock_mentions,
    generate_financial_term_answer_stream,
    generate_follow_up_answer_stream,
    generate_investment_advice_stream,
    generate_portfolio_analysis_stream,
    resolve_tw_company_name,
    generate_user_news_sentiment_answer_stream,
    run_quant_model,
)
from sentiment_analysis import load_finbert_model


APP_DIR = Path(__file__).resolve().parent
USER_HISTORY_DIR = APP_DIR / "data" / "user_histories"
DEFAULT_ASSISTANT_MESSAGE = (
    "您好！我是您的台股投資顧問，您可以直接輸入標的，例如：「台積電走勢如何？」"
)
DEFAULT_PROFILE = {"style": "穩健", "risk_tolerance": "medium", "max_loss_pct": 10}
RISK_MAP = {"保守": "low", "穩健": "medium", "積極": "high"}


class InvestorProfile(BaseModel):
    style: str = "穩健"
    risk_tolerance: str = "medium"
    max_loss_pct: int = Field(default=10, ge=1, le=80)


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    role: str
    content: str
    created_at: float = Field(default_factory=time.time)
    dashboard_data: Optional[Dict[str, Any]] = None


class SessionSummary(BaseModel):
    session_id: str
    title: str
    messages: List[ChatMessage]


class AppState(BaseModel):
    user_id: str
    sessions: Dict[str, List[ChatMessage]]
    current_session: str
    session_counter: int
    investor_profile: InvestorProfile


class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = None


class StreamMessageRequest(BaseModel):
    user_id: Optional[str] = None
    content: str
    profile: InvestorProfile = Field(default_factory=InvestorProfile)
    model: Optional[str] = None


class ProfileRequest(BaseModel):
    user_id: str
    profile: InvestorProfile


app = FastAPI(title="Taiwan Stock Advisor API")


def _env_csv(name: str) -> List[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


_default_dev_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

cors_allow_origins = _env_csv("CORS_ALLOW_ORIGINS") or _default_dev_origins
cors_allow_origin_regex = os.getenv("CORS_ALLOW_ORIGIN_REGEX", "").strip() or r"^https://.*\.vercel\.app$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_origin_regex=cors_allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _preload_finbert():
    """Pre-warm the FinBERT model at startup so the first user query is fast."""
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_finbert_model)
    print("\u2705 FinBERT \u6a21\u578b\u9810\u71b1\u5b8c\u6210\uff0c\u7cfb\u7d71\u5c31\u7dd2")


def _safe_user_id(user_id: Optional[str]) -> str:
    if isinstance(user_id, str) and re.fullmatch(r"[a-zA-Z0-9_-]{8,64}", user_id):
        return user_id
    return uuid.uuid4().hex


def _chat_store_path(user_id: str) -> Path:
    safe_id = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]
    return USER_HISTORY_DIR / f"{safe_id}.json"


def _session_title(messages: List[ChatMessage]) -> str:
    for message in messages:
        if message.role == "user" and message.content.strip():
            limit = 30
            return message.content[:limit] + ("..." if len(message.content) > limit else "")
    return "新對話"


def _df_to_records(df_history: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(df_history, pd.DataFrame) or df_history.empty:
        return None
    df = df_history.copy().tail(252)
    df.index = df.index.astype(str)
    records = df.reset_index(names="date").to_dict(orient="records")
    return json.loads(json.dumps(records, default=str, ensure_ascii=False))


def _clean_dashboard_data(dashboard_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not dashboard_data:
        return None
    payload = dict(dashboard_data)
    payload["chart_data"] = _df_to_records(payload.pop("df_history", None))
    return json.loads(json.dumps(payload, default=str, ensure_ascii=False))


def _load_state(user_id: str) -> AppState:
    path = _chat_store_path(user_id)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            sessions = {
                session_id: [ChatMessage(**message) for message in messages]
                for session_id, messages in payload.get("sessions", {}).items()
            }
            if sessions:
                current_session = payload.get("current_session") or next(iter(sessions))
                if current_session not in sessions:
                    current_session = next(iter(sessions))
                return AppState(
                    user_id=user_id,
                    sessions=sessions,
                    current_session=current_session,
                    session_counter=int(payload.get("session_counter", len(sessions))),
                    investor_profile=InvestorProfile(**payload.get("investor_profile", DEFAULT_PROFILE)),
                )
        except Exception:
            pass

    return AppState(
        user_id=user_id,
        sessions={"session_1": [ChatMessage(role="assistant", content=DEFAULT_ASSISTANT_MESSAGE)]},
        current_session="session_1",
        session_counter=1,
        investor_profile=InvestorProfile(**DEFAULT_PROFILE),
    )


def _save_state(state: AppState) -> None:
    USER_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "sessions": {
            session_id: [message.model_dump() for message in messages]
            for session_id, messages in state.sessions.items()
        },
        "current_session": state.current_session,
        "session_counter": state.session_counter,
        "investor_profile": state.investor_profile.model_dump(),
    }
    with _chat_store_path(state.user_id).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _normalize_ticker(value: Any) -> str:
    if not value:
        return ""
    return str(value).upper().replace(".TW", "").replace(".TWO", "").strip()


def _same_stock_follow_up(question: str, dashboard_data: Optional[Dict[str, Any]]) -> bool:
    if not dashboard_data:
        return False
    # If 2+ ticker-like numbers appear, treat as a new compare query, not a follow-up
    all_tickers = re.findall(r"(?<!\d)(\d{4})(?!\d)", question)
    if len(all_tickers) >= 2:
        return False
    ticker = _normalize_ticker(dashboard_data.get("ticker"))
    company_name = str(dashboard_data.get("company_name", ""))
    # Check numeric ticker matches
    if all_tickers:
        return ticker in all_tickers
    # Fallback: check if the company name is in the question
    if company_name and company_name not in ["未知", "大盤"]:
        return company_name in question
    return False


def _latest_dashboard(messages: List[ChatMessage]) -> Optional[Dict[str, Any]]:
    for message in reversed(messages):
        if message.dashboard_data:
            return message.dashboard_data
    return None


def _normalize_stock_mentions(stock_mentions: List[Any]) -> List[Dict[str, str]]:
    normalized = []
    for stock in stock_mentions:
        if isinstance(stock, dict):
            ticker = stock.get("ticker")
            company_name = stock.get("company_name") or ticker
        else:
            ticker = str(stock)
            company_name = ticker
        if ticker:
            normalized.append({"ticker": ticker, "company_name": company_name})
    return normalized


def _analyze_stock_for_comparison(stock_info: Dict[str, str]) -> Dict[str, Any]:
    ticker = stock_info.get("ticker")
    company_name = stock_info.get("company_name") or ticker
    stock_data, df_history = fetch_realtime_stock_data(ticker)
    resolved_ticker = stock_data.get("resolved_ticker") if stock_data else None
    display_ticker = resolved_ticker or _normalize_ticker(ticker) or ticker
    company_name = resolve_tw_company_name(str(display_ticker or ticker or ""), str(company_name or ""))
    news_data = fetch_stock_or_macro_sentiment(display_ticker, company_name, days=5)
    quant_data = run_quant_model(display_ticker, df_history, {
        "ticker": display_ticker,
        "company_name": company_name,
    })
    return {
        "標的": f"{company_name}（{display_ticker}）",
        "目前價格": stock_data.get("latest_price", "未知") if stock_data else "未知",
        "趨勢": stock_data.get("trend_summary", "未知") if stock_data else "未知",
        "量化訊號": quant_data.get("signal", "未知"),
        "預期報酬率": f"{quant_data.get('predicted_return', 0) * 100:.3f}%",
        "情緒分數": news_data.get("sentiment_score", 0),
        "5天新聞數": len(news_data.get("raw_news", [])),
        "市場狀態": quant_data.get("regime", "未知"),
        "近一年最大回撤": quant_data.get("max_dd", "N/A"),
    }

def _stock_lookup_failed(ticker: Any, stock_data: Optional[Dict[str, Any]], df_history: Any) -> bool:
    normalized = _normalize_ticker(ticker)
    if not normalized or normalized in ["未知", "未知標的"]:
        return True
    if df_history is None:
        return True
    if isinstance(df_history, pd.DataFrame):
        return df_history.empty
    # Fallback: if the fetcher returned some history-like object, treat it as success.
    return False


def _format_portfolio_table(rows: List[Dict[str, Any]]) -> str:
    columns = list(rows[0].keys()) if rows else []
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def _event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _stream_text(stream: Generator[str, None, None]) -> Generator[str, None, str]:
    chunks = []
    for chunk in stream:
        chunks.append(chunk)
        yield _event("token", {"text": chunk})
    return "".join(chunks)


def _run_analysis(
    state: AppState,
    session_id: str,
    user_input: str,
    profile: InvestorProfile,
    api_key: Optional[str],
    model: Optional[str],
) -> Generator[str, None, None]:
    messages = state.sessions[session_id]
    user_message = ChatMessage(role="user", content=user_input)
    messages.append(user_message)
    state.investor_profile = profile
    _save_state(state)
    yield _event("message", {"message": user_message.model_dump()})

    final_reply = ""
    dashboard_payload = None

    try:
        with groq_request_context(api_key=api_key, model=model):
            previous_dashboard = _latest_dashboard(messages[:-1])
            if previous_dashboard and _same_stock_follow_up(user_input, previous_dashboard):
                yield _event("status", {"text": "沿用上一份分析資料回答延伸問題..."})
                final_reply = yield from _stream_text(
                    generate_follow_up_answer_stream(user_input, previous_dashboard, [m.model_dump() for m in messages])
                )
            else:
                yield _event("status", {"text": "解析投資意圖與標的..."})
                intent = extract_intent(user_input, 1, profile.risk_tolerance, [m.model_dump() for m in messages])
                intent["risk_tolerance"] = profile.risk_tolerance
                intent["investor_profile"] = profile.model_dump()

                if intent.get("intent_type") == "UNRELATED":
                    final_reply = "不好意思，請輸入與投資有關的問題喔！我是專注於台股量化與情緒分析的 AI 投資顧問。"
                    yield _event("token", {"text": final_reply})
                elif intent.get("intent_type") == "FINANCIAL_TERM":
                    yield _event("status", {"text": "辨識為金融術語說明..."})
                    final_reply = yield from _stream_text(generate_financial_term_answer_stream(user_input, profile.model_dump()))
                elif intent.get("intent_type") == "USER_NEWS":
                    yield _event("status", {"text": "分析使用者提供的新聞情緒..."})
                    analysis = analyze_user_provided_news(user_input)
                    final_reply = yield from _stream_text(
                        generate_user_news_sentiment_answer_stream(user_input, analysis, profile.model_dump())
                    )
                else:
                    stock_mentions = _normalize_stock_mentions(intent.get("stocks") or find_stock_mentions(user_input))
                    wants_portfolio = len(stock_mentions) > 1 and any(
                        word in user_input for word in ["手上", "現有", "持股", "健檢", "投資組合", "看看", "這些", "這幾檔"]
                    )
                    # Any 2+ stocks mentioned => compare table, even without explicit compare keywords
                    wants_compare = not wants_portfolio and len(stock_mentions) >= 2

                    if wants_portfolio or wants_compare:
                        limit = 5 if wants_portfolio else 2
                        rows = []
                        for idx, stock_info in enumerate(stock_mentions[:limit], start=1):
                            yield _event("status", {"text": f"分析第 {idx} 檔：{stock_info.get('company_name')}..."})
                            rows.append(_analyze_stock_for_comparison(stock_info))
                        table_md = _format_portfolio_table(rows)
                        heading = "投資組合/持股健檢結果" if wants_portfolio else "兩檔標的比較"
                        yield _event("token", {"text": f"### {heading}\n\n{table_md}\n\n"})
                        analysis = yield from _stream_text(
                            generate_portfolio_analysis_stream(
                                rows,
                                user_input,
                                profile.model_dump(),
                                intent_type="PORTFOLIO" if wants_portfolio else "COMPARE",
                            )
                        )
                        final_reply = f"### {heading}\n\n{table_md}\n\n{analysis}"
                    else:
                        ticker = intent.get("ticker")
                        company_name = intent.get("company_name", "未知")

                        t0 = time.time()
                        yield _event("status", {"text": f"[1/4] 📡 抓取 {company_name}({ticker}) 即時行情..."})
                        stock_data, df_history = fetch_realtime_stock_data(ticker)
                        if stock_data and stock_data.get("resolved_ticker"):
                            ticker = stock_data["resolved_ticker"]
                        company_name = resolve_tw_company_name(str(ticker or ""), str(company_name or ""))
                        t1 = time.time()

                        if _stock_lookup_failed(ticker, stock_data, df_history):
                            normalized = _normalize_ticker(ticker) or str(ticker or "").strip() or "未知"
                            final_reply = f"查無 {normalized} 對應的台股標的，請確認股票代號或公司名稱後再試。"
                            yield _event("token", {"text": final_reply})
                            assistant_message = ChatMessage(role="assistant", content=final_reply)
                            messages.append(assistant_message)
                            _save_state(state)
                            yield _event("done", {"message": assistant_message.model_dump()})
                            return

                        yield _event("status", {"text": f"[2/4] 📰 掃描近期新聞 & 情緒分析... ({round(t1-t0,1)}s)"})
                        news_data = fetch_stock_or_macro_sentiment(ticker, company_name, days=5)
                        t2 = time.time()

                        yield _event("status", {"text": f"[3/4] 🤖 量化模型運算中... ({round(t2-t1,1)}s)"})
                        quant_data = run_quant_model(ticker, df_history, intent)
                        t3 = time.time()

                        if ticker not in ["未知", "未知標的", None]:
                            dashboard_payload = _clean_dashboard_data({
                                "ticker": _normalize_ticker(ticker),
                                "company_name": resolve_tw_company_name(str(ticker or ""), str(company_name or "")),
                                "stock_data": stock_data,
                                "news_data": news_data,
                                "quant_data": quant_data,
                                "df_history": df_history,
                                "risk": profile.risk_tolerance,
                                "investor_profile": profile.model_dump(),
                            })
                            yield _event("dashboard", {"dashboard_data": dashboard_payload})

                        yield _event("status", {"text": f"[4/4] ✍️ 生成分析報告... ({round(t3-t2,1)}s · 總耗時 {round(t3-t0,1)}s)"})
                        final_reply = yield from _stream_text(
                            generate_investment_advice_stream(
                                user_input,
                                intent,
                                quant_data,
                                stock_data,
                                news_data,
                                [m.model_dump() for m in messages],
                                profile.model_dump(),
                            )
                        )

        assistant_message = ChatMessage(role="assistant", content=final_reply, dashboard_data=dashboard_payload)
        messages.append(assistant_message)
        _save_state(state)
        yield _event("done", {"message": assistant_message.model_dump()})
    except Exception as exc:
        if hasattr(exc, "response") and exc.response is not None and exc.response.status_code == 429:
            error_message = "API 請求次數過多 (Too Many Requests)，請稍後再試。"
        elif "429 Client Error" in str(exc):
            error_message = "API 請求次數過多 (Too Many Requests)，請稍後再試。"
        else:
            # 避免直接顯示含有網址的錯誤訊息
            error_detail = str(exc)
            if "https://" in error_detail or "http://" in error_detail:
                import re
                error_detail = re.sub(r'https?://[^\s]+', '<API Endpoint>', error_detail)
            error_message = f"分析時發生錯誤：{type(exc).__name__}: {error_detail}"
        
        assistant_message = ChatMessage(role="assistant", content=error_message)
        messages.append(assistant_message)
        _save_state(state)
        yield _event("error", {"message": error_message})


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "default_model": config.GROQ_MODEL}


@app.get("/api/sessions")
def list_sessions(user_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    safe_user_id = _safe_user_id(user_id)
    state = _load_state(safe_user_id)
    _save_state(state)
    sessions = [
        SessionSummary(session_id=session_id, title=_session_title(messages), messages=messages).model_dump()
        for session_id, messages in state.sessions.items()
    ]
    return {
        "user_id": safe_user_id,
        "current_session": state.current_session,
        "investor_profile": state.investor_profile.model_dump(),
        "sessions": sessions,
    }


@app.post("/api/sessions")
def create_session(request: CreateSessionRequest) -> Dict[str, Any]:
    safe_user_id = _safe_user_id(request.user_id)
    state = _load_state(safe_user_id)
    state.session_counter += 1
    session_id = f"session_{state.session_counter}"
    state.sessions[session_id] = [ChatMessage(role="assistant", content="您好！這是一個新的分析對話，請問今天想了解哪一檔股票？")]
    state.current_session = session_id
    _save_state(state)
    return {"user_id": safe_user_id, "session_id": session_id, "messages": [m.model_dump() for m in state.sessions[session_id]]}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, user_id: str = Query(...)) -> Dict[str, Any]:
    state = _load_state(_safe_user_id(user_id))
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    state.current_session = session_id
    _save_state(state)
    return {"session_id": session_id, "messages": [m.model_dump() for m in state.sessions[session_id]]}


@app.patch("/api/profile")
def update_profile(request: ProfileRequest) -> Dict[str, Any]:
    state = _load_state(_safe_user_id(request.user_id))
    profile = request.profile
    profile.risk_tolerance = RISK_MAP.get(profile.style, profile.risk_tolerance)
    state.investor_profile = profile
    _save_state(state)
    return {"investor_profile": state.investor_profile.model_dump()}


@app.post("/api/sessions/{session_id}/messages/stream")
def stream_message(
    session_id: str,
    request: StreamMessageRequest,
    x_groq_api_key: Optional[str] = Header(default=None),
) -> StreamingResponse:
    safe_user_id = _safe_user_id(request.user_id)
    state = _load_state(safe_user_id)
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    state.current_session = session_id
    request.profile.risk_tolerance = RISK_MAP.get(request.profile.style, request.profile.risk_tolerance)
    return StreamingResponse(
        _run_analysis(state, session_id, request.content.strip(), request.profile, x_groq_api_key, request.model),
        media_type="text/event-stream",
    )
