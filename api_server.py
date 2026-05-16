import hashlib
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio

import config
from config import groq_request_context
from data_fetch import (
    analyze_user_provided_news,
    extract_intent,
    fetch_realtime_stock_data,
    fetch_stock_or_macro_sentiment,
    find_stock_mentions,
    FINANCIAL_TERMS,
    UNRELATED_KEYWORDS,
    generate_financial_term_answer_stream,
    generate_follow_up_answer_stream,
    generate_investment_advice_stream,
    generate_portfolio_analysis_stream,
    resolve_tw_company_name,
    generate_user_news_sentiment_answer_stream,
    run_quant_model,
)


APP_DIR = Path(__file__).resolve().parent
USER_HISTORY_DIR = APP_DIR / "data" / "user_histories"
DEFAULT_ASSISTANT_MESSAGE = (
    "您好！我是您的台股投資顧問，您可以直接輸入標的，例如：「台積電走勢如何？」"
)

T = TypeVar("T")


class StepTimeoutError(Exception):
    def __init__(self, step: str, seconds: int):
        self.step = step
        self.seconds = seconds
        super().__init__(f"{step} timed out after {seconds}s")


def _run_with_timeout(step: str, seconds: int, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        raise StepTimeoutError(step, seconds) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
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
    analysis_context: Optional[Dict[str, Any]] = None


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

@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok"}


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
local_cors_regex = r"^http://(localhost|127\.0\.0\.1):[0-9]+$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_origin_regex=local_cors_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def _preload_finbert():
    """Pre-warm the FinBERT model for the local-only app."""
    if os.getenv("PRELOAD_FINBERT", "").strip() not in {"1", "true", "True", "yes", "YES"}:
        print("FinBERT 預熱已停用；小型雲端主機會使用首次請求載入或關鍵字備援。")
        return
    try:
        from sentiment_analysis import warm_finbert_model

        await asyncio.to_thread(warm_finbert_model)
        print("FinBERT 模型預熱完成。")
    except Exception as exc:
        print(f"FinBERT 預熱失敗，將於首次情緒分析時再嘗試載入：{exc}")


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
    return str(value).upper().replace(".TWO", "").replace(".TW", "").strip()


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


def _latest_analysis_context(messages: List[ChatMessage]) -> Optional[Dict[str, Any]]:
    for message in reversed(messages):
        if message.analysis_context:
            return message.analysis_context
        if message.dashboard_data:
            context = dict(message.dashboard_data)
            context.setdefault("mode", "stock")
            return context
    return None


def _latest_user_message(messages: List[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    return ""


def _looks_like_fresh_analysis(question: str) -> bool:
    text = (question or "").strip()
    if not text:
        return False
    # "Fresh analysis" should be used only when the user explicitly mentions a target.
    has_explicit_target = bool(find_stock_mentions(text) or re.search(r"(?<!\d)\d{4,6}(?!\d)", text))
    if not has_explicit_target:
        return False
    return any(keyword in text for keyword in ["分析", "現在", "位階", "適合", "可以買", "值得買", "走勢", "進場", "幫我看", "想關注"])


def _is_refresh_request(question: str) -> bool:
    text = (question or "").strip()
    return any(keyword in text for keyword in ["重新分析", "重新抓", "重跑", "再算一次", "更新資料", "最新資料", "重新整理"])


def _context_targets(context: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    if not context:
        return []
    if context.get("mode") == "compare":
        stocks = context.get("stocks") or []
        if isinstance(stocks, list):
            return _normalize_stock_mentions(stocks)
        return []
    ticker = _normalize_ticker(context.get("ticker"))
    company_name = str(context.get("company_name") or ticker or "")
    return [{"ticker": ticker, "company_name": company_name}] if ticker or company_name else []


def _mentions_different_target(question: str, context: Optional[Dict[str, Any]]) -> bool:
    targets = _context_targets(context)
    if not targets:
        return False
    context_tickers = {_normalize_ticker(item.get("ticker")) for item in targets if item.get("ticker")}
    context_names = {str(item.get("company_name") or "") for item in targets if item.get("company_name")}
    mentioned = find_stock_mentions(question) or []
    numeric_codes = re.findall(r"(?<!\d)(\d{4,6})(?!\d)", question or "")
    for code in numeric_codes:
        if code and code not in context_tickers:
            return True
    for item in mentioned:
        ticker = _normalize_ticker(item.get("ticker"))
        name = str(item.get("company_name") or "")
        if ticker and ticker not in context_tickers:
            return True
        if name and name not in context_names:
            return True
    return False


def _mentions_context_target(question: str, context: Optional[Dict[str, Any]]) -> bool:
    targets = _context_targets(context)
    if not targets:
        return False
    text = question or ""
    context_tickers = {_normalize_ticker(item.get("ticker")) for item in targets if item.get("ticker")}
    numeric_codes = set(re.findall(r"(?<!\d)(\d{4,6})(?!\d)", text))
    if context_tickers.intersection(numeric_codes):
        return True
    for item in targets:
        name = str(item.get("company_name") or "")
        if name and name not in ["未知", "大盤"] and name in text:
            return True
    return False


def _is_context_follow_up(question: str, context: Optional[Dict[str, Any]]) -> bool:
    if not context or _is_refresh_request(question) or _mentions_different_target(question, context):
        return False
    text = (question or "").strip()
    if not text:
        return False
    if _mentions_context_target(text, context):
        return True
    followup_triggers = [
        "這篇",
        "剛剛",
        "你剛剛",
        "剛才",
        "會有影響",
        "這對",
        "那這樣",
        "建議",
        "進場",
        "停損",
        "買入訊號",
        "可信度",
        "預期報酬",
        "只能選",
        "優先",
        "哪一檔",
        "CP",
        "風險",
        "配置",
        "分配",
        "外資",
        "庫存",
        "提款",
        "修正",
        "壓力",
        "利空",
        "利多",
        "新聞",
    ]
    if any(keyword in text for keyword in followup_triggers):
        return True
    if any(term.lower() in text.lower() for term in FINANCIAL_TERMS):
        return True
    return False


def _looks_like_context_follow_up(question: str, dashboard_data: Optional[Dict[str, Any]]) -> bool:
    """
    Treat questions like "這篇新聞會影響你剛剛的建議嗎？" as follow-ups to the latest dashboard,
    even if the user doesn't repeat the company name/ticker.
    """
    if not dashboard_data:
        return False
    text = (question or "").strip()
    if not text:
        return False

    # If the user explicitly mentions a *different* target, do not treat as follow-up.
    mentions = find_stock_mentions(text) or []
    numeric_codes = re.findall(r"(?<!\d)(\d{4,6})(?!\d)", text)
    prev_ticker = _normalize_ticker(dashboard_data.get("ticker"))
    prev_company = str(dashboard_data.get("company_name", "") or "")
    prev_company = prev_company if prev_company not in ["未知", "大盤"] else ""

    # Any explicit mention of a different ticker => new query.
    if numeric_codes:
        return prev_ticker in numeric_codes and len(set(numeric_codes)) == 1
    if mentions:
        # If user mentions any stock name/ticker not matching previous, treat as new query.
        for m in mentions:
            t = _normalize_ticker(m.get("ticker"))
            n = str(m.get("company_name") or "")
            if t and prev_ticker and t != prev_ticker:
                return False
            if n and prev_company and n != prev_company:
                return False
        return True

    followup_triggers = [
        "這篇",
        "剛剛",
        "你剛剛",
        "剛才",
        "會有影響嗎",
        "會影響嗎",
        "這對",
        "那這樣",
        "建議",
        "進場",
        "停損",
        "外資",
        "庫存",
        "提款",
        "修正",
        "壓力",
        "利空",
        "利多",
        "新聞",
    ]
    if any(t in text for t in followup_triggers):
        return True
    # Financial term follow-up without a target: tie back to the previous dashboard.
    if any(term.lower() in text.lower() for term in FINANCIAL_TERMS):
        return True
    return False


def _compare_rank_markdown(rows: List[Dict[str, Any]]) -> str:
    """
    Deterministic post-table recommendation so compare results always include a clear suggestion.
    Rules:
      - signal: 買入 > 觀望 > 賣出 > 無法預測/未知
      - then predicted return (higher better)
      - then max drawdown (smaller absolute better)
      - data-insufficient goes to a separate section
    """
    if not rows:
        return ""

    def parse_signal(value: Any) -> int:
        s = str(value or "")
        if "買入" in s:
            return 3
        if "觀望" in s:
            return 2
        if "賣出" in s:
            return 1
        return 0

    def parse_pct(value: Any) -> float:
        s = str(value or "").strip()
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else 0.0

    def parse_dd_abs(value: Any) -> float:
        s = str(value or "")
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if not m:
            return 999.0
        return abs(float(m.group(0)))

    def label(row: Dict[str, Any]) -> str:
        return str(row.get("標的") or "")

    ranked = []
    insufficient = []
    for row in rows:
        sig = str(row.get("量化訊號") or "")
        if "抓取失敗" in sig or "無法預測" in sig or "無資料" in sig:
            insufficient.append(row)
            continue
        ranked.append(
            (
                parse_signal(sig),
                parse_pct(row.get("預期報酬率")),
                -parse_dd_abs(row.get("近一年最大回撤")),
                row,
            )
        )

    ranked.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    picks = [t[3] for t in ranked]

    lines: List[str] = []
    if picks:
        primary = label(picks[0])
        secondary = label(picks[1]) if len(picks) > 1 else ""
        avoid = label(picks[-1]) if len(picks) > 2 else ""
        lines.append("### 建議（依量化訊號優先）")
        lines.append(f"- 首選：{primary}")
        if secondary:
            lines.append(f"- 次選：{secondary}")
        if avoid and avoid not in [primary, secondary]:
            lines.append(f"- 相對保守：{avoid}")
        lines.append("- 配置建議：以首選為主、次選為輔，資料不足者暫不納入核心配置。")
    if insufficient:
        lines.append("")
        lines.append("### 資料不足/暫不排名")
        for row in insufficient:
            lines.append(f"- {label(row)}：目前資料不足或抓取失敗")
    return "\n".join(lines).strip()


def _build_compare_analysis_context(
    rows: List[Dict[str, Any]],
    stock_mentions: List[Dict[str, str]],
    profile: InvestorProfile,
    heading: str,
    compared_names: str,
    recommendation: str,
) -> Dict[str, Any]:
    return json.loads(json.dumps({
        "mode": "compare",
        "ticker": "COMPARE",
        "company_name": "多檔比較",
        "heading": heading,
        "compared_names": compared_names,
        "stocks": stock_mentions,
        "compare_rows": rows,
        "recommendation": recommendation,
        "risk": profile.risk_tolerance,
        "investor_profile": profile.model_dump(),
    }, ensure_ascii=False, default=str))


def _normalize_stock_mentions(stock_mentions: List[Any]) -> List[Dict[str, str]]:
    normalized = []
    seen = set()
    for stock in stock_mentions:
        if isinstance(stock, dict):
            ticker = stock.get("ticker")
            company_name = stock.get("company_name") or ticker
        else:
            ticker = str(stock)
            company_name = ticker
        if ticker:
            key = _normalize_ticker(ticker)
            if key in seen:
                continue
            seen.add(key)
            normalized.append({"ticker": ticker, "company_name": company_name})
    return normalized


def _merge_stock_mentions(*groups: List[Any]) -> List[Dict[str, str]]:
    merged: List[Any] = []
    for group in groups:
        if group:
            merged.extend(group)
    return _normalize_stock_mentions(merged)


def _analyze_stock_for_comparison(stock_info: Dict[str, str]) -> Dict[str, Any]:
    ticker = stock_info.get("ticker")
    company_name = stock_info.get("company_name") or ticker
    try:
        stock_data, df_history = _run_with_timeout("股價資料抓取", 20, fetch_realtime_stock_data, ticker)
    except StepTimeoutError:
        stock_data, df_history = None, None
    resolved_ticker = stock_data.get("resolved_ticker") if stock_data else None
    display_ticker = resolved_ticker or _normalize_ticker(ticker) or ticker
    company_name = resolve_tw_company_name(str(display_ticker or ticker or ""), str(company_name or ""))
    try:
        news_data = _run_with_timeout("新聞抓取", 25, fetch_stock_or_macro_sentiment, display_ticker, company_name, days=5)
    except StepTimeoutError:
        news_data = _fallback_news_data("新聞抓取失敗：新聞來源回應逾時，情緒分數先以 0 處理。")
    except Exception:
        news_data = _fallback_news_data("新聞抓取失敗：本地新聞來源暫時無法取得，情緒分數先以 0 處理。")
    if _stock_lookup_failed(display_ticker, stock_data, df_history):
        quant_data = _fallback_quant_data("行情或歷史資料不足")
    else:
        try:
            quant_data = _run_with_timeout("量化模型", 35, run_quant_model, display_ticker, df_history, {
                "ticker": display_ticker,
                "company_name": company_name,
            })
        except StepTimeoutError:
            quant_data = _fallback_quant_data("量化模型回應逾時")
        except Exception:
            quant_data = _fallback_quant_data("本地量化模型載入失敗")
    latest_price = stock_data.get("latest_price", "未知") if stock_data else "未知"
    predicted_return = float(quant_data.get("predicted_return", 0) or 0)
    expected_move = "未知"
    if isinstance(latest_price, (int, float)):
        expected_move = f"{float(latest_price) * predicted_return:.2f}"
    return {
        "標的": f"{company_name}（{display_ticker}）",
        "目前價格": latest_price,
        "趨勢": stock_data.get("trend_summary", "未知") if stock_data else "未知",
        "量化訊號": quant_data.get("signal", "未知"),
        "預期報酬率": f"{quant_data.get('predicted_return', 0) * 100:.3f}%",
        "預期價差": expected_move,
        "情緒分數": news_data.get("sentiment_score", 0),
        "5天新聞數": "抓取失敗" if news_data.get("news_count_status") == "fetch_failed" else len(news_data.get("raw_news", [])),
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


def _fallback_news_data(message: str = "新聞抓取失敗：目前無法取得近期新聞，情緒分數先以 0 處理。") -> Dict[str, Any]:
    return {
        "news_summary": message,
        "sentiment_score": 0.0,
        "raw_news": [],
        "news_count_status": "fetch_failed",
    }


def _fallback_quant_data(reason: str = "量化模型暫時無法完成預測") -> Dict[str, Any]:
    return {
        "signal": "無法預測",
        "predicted_return": 0.0,
        "predicted_move": "未知",
        "regime": "無資料",
        "max_dd": "N/A",
        "model_name": reason,
    }


def _fallback_intent(question: str, profile: InvestorProfile) -> Dict[str, Any]:
    mentions = find_stock_mentions(question)
    if mentions:
        first = mentions[0]
        return {
            "intent_type": "STOCK",
            "ticker": first.get("ticker"),
            "company_name": first.get("company_name"),
            "stocks": mentions,
            "risk_tolerance": profile.risk_tolerance,
            "investor_profile": profile.model_dump(),
        }
    return {
        "intent_type": "UNRELATED",
        "risk_tolerance": profile.risk_tolerance,
        "investor_profile": profile.model_dump(),
    }


def _friendly_error_message(exc: Exception) -> str:
    if isinstance(exc, StepTimeoutError):
        if "新聞" in exc.step:
            return "新聞抓取失敗：新聞來源回應逾時，請稍後再試。"
        if "股價" in exc.step or "行情" in exc.step:
            return "有標的但資料不足：行情來源回應逾時，暫時無法完成分析。"
        return f"{exc.step}逾時：本地後端仍在處理或資料來源回應過慢，請稍後再試。"
    if hasattr(exc, "response") and exc.response is not None and exc.response.status_code == 429:
        return "API 請求次數過多 (Too Many Requests)，請稍後再試。"
    if "429 Client Error" in str(exc):
        return "API 請求次數過多 (Too Many Requests)，請稍後再試。"
    text = str(exc)
    if "NoneType" in text or isinstance(exc, TypeError):
        return "有標的但資料不足：部分資料來源暫時沒有回傳可用內容，請稍後再試。"
    if "Google News" in text or "RSS" in text or "新聞" in text:
        return "新聞抓取失敗：目前無法取得近期新聞，請稍後再試。"
    return "分析時發生錯誤：本地資料來源暫時不穩定，請稍後再試。"


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


def sanitize_assistant_text(text: str) -> str:
    if not text:
        return ""
    blocked_patterns = [
        r"情況\s*[A-DＡ-Ｄ]",
        r"Case\s*[A-D]",
        r"intent_type",
        r"問題.{0,12}(類型|分類|模式)",
        r"(類型|分類|模式).{0,12}問題",
        r"這是.{0,12}(類型|分類|模式)",
        r"屬於.{0,12}(類型|分類|模式)",
    ]
    kept_lines = []
    for line in str(text).splitlines():
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in blocked_patterns):
            continue
        kept_lines.append(line)
    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _looks_like_unrelated_local(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if any(kw in t for kw in UNRELATED_KEYWORDS):
        return True
    # Very short utterances are almost always small talk; avoid stock lookup attempts.
    if len(t) <= 4 and not re.search(r"(?<!\d)\d{4}(?!\d)", t):
        return True
    return False


def _looks_like_financial_term_only(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    has_term = any(term.lower() in t.lower() for term in FINANCIAL_TERMS)
    if not has_term:
        return False
    if find_stock_mentions(t):
        return False
    if re.search(r"(?<!\d)(\d{4})(?:\.(TW|TWO))?(?!\d)", t, re.IGNORECASE):
        return False
    return True


def _stream_text(stream: Generator[str, None, None]) -> Generator[str, None, str]:
    chunks: List[str] = []
    for chunk in stream:
        chunks.append(chunk)
    text = sanitize_assistant_text("".join(chunks))
    text = _ensure_complete_assistant_text(text)
    for start in range(0, len(text), 600):
        yield _event("token", {"text": text[start:start + 600]})
    return text


def _ensure_complete_assistant_text(text: str) -> str:
    cleaned = (text or "").rstrip()
    if not cleaned:
        return cleaned

    lines = [line.rstrip() for line in cleaned.splitlines()]
    last = next((line.strip() for line in reversed(lines) if line.strip()), "")
    dangling_warning_patterns = [
        r"^>?\s*⚠️?\s*本報告\s*$",
        r"^>?\s*⚠️?\s*本報告基於\s*$",
        r"^>?\s*⚠️?\s*本報告基於量化模型與新聞情緒模型\s*$",
        r"^>?\s*⚠️?\s*本報告基於量化模型與新聞情緒模型即時運算\s*[，,]?\s*$",
    ]
    has_dangling_warning = any(re.search(pattern, last) for pattern in dangling_warning_patterns)
    if has_dangling_warning:
        # Drop the partially generated disclaimer line before appending a clean close.
        for index in range(len(lines) - 1, -1, -1):
            if lines[index].strip() == last:
                lines = lines[:index]
                break
        cleaned = "\n".join(lines).rstrip()
        last = next((line.strip() for line in reversed(lines) if line.strip()), "")

    unfinished_patterns = [
        r"^[-•o○]$",
        r"^[-•o○]\s*[\u4e00-\u9fffA-Za-z]{1,2}$",
        r"^\d+[.、]$",
        r"^\d+[.、]\s*[\u4e00-\u9fffA-Za-z0-9 ／/：:（）()]{1,18}$",
        r"^#{1,6}\s*[\u4e00-\u9fffA-Za-z0-9 ／/：:（）()]{1,18}$",
        r"^[\u4e00-\u9fffA-Za-z0-9 ／/：:（）()]{1,18}[：:]$",
        r"^(風險管理|停損設定|進場方式|配置建議|操作建議|觀察重點)$",
        r"^⚠️?\s*本報告.*$",
    ]
    looks_unfinished = any(re.search(pattern, last) for pattern in unfinished_patterns)
    ends_without_sentence = not re.search(r"[。！？.!?）)]$", last)
    has_timeout_note = "生成時間較長" in cleaned or "目前可用重點" in cleaned
    if looks_unfinished or has_timeout_note or has_dangling_warning or ends_without_sentence:
        return (
            cleaned
            + "\n\n### 完整收尾\n\n"
            + "- **先不要把單一訊號當成唯一依據**：請把量化訊號、股價趨勢、回撤風險與新聞情緒一起看。\n"
            + "- **操作上採保守分批**：若原本想一次進場，建議先降低單次部位，等股價站穩關鍵價位或量能確認後再加碼。\n"
            + "- **停損要先寫好**：停損點以你的最大可接受虧損為上限；若價格跌破重要支撐或量化訊號轉弱，應優先控風險。\n\n"
            + "> 本分析僅供研究與決策輔助參考，不構成投資建議。"
        )
    return cleaned


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
    yield _event("status", {"text": "已收到請求，正在啟動分析流程..."})
    yield _event("message", {"message": user_message.model_dump()})

    final_reply = ""
    dashboard_payload = None
    analysis_context = None
    early_handled = False

    try:
        with groq_request_context(api_key=api_key, model=model):
            previous_context = _latest_analysis_context(messages[:-1])
            previous_user_message = _latest_user_message(messages[:-1])
            is_repeat_question = previous_user_message == user_input.strip()

            # Rule-first hard guardrails (run before any follow-up / intent parsing):
            # 1) Unrelated / small talk: do not attempt stock lookup.
            if _looks_like_unrelated_local(user_input):
                final_reply = "請輸入跟台股投資、金融術語或市場分析相關的問題。"
                yield _event("token", {"text": final_reply})
                analysis_context = None
                early_handled = True

            # 2) Financial term only (no ticker/company): always answer the term.
            # If we have a previous analysis context, we explain the term in that context.
            if not early_handled and _looks_like_financial_term_only(user_input):
                yield _event("status", {"text": "辨識為金融術語說明..."})
                # Term-only questions must be explained as education content.
                # Do not route them through stock follow-up context, otherwise
                # simple questions like "MACD 是什麼" can become empty or too narrow.
                final_reply = yield from _stream_text(
                    generate_financial_term_answer_stream(user_input, profile.model_dump())
                )
                analysis_context = None
                early_handled = True

            if early_handled:
                # Fall through to persist the assistant message and emit the done event.
                raise RuntimeError("__EARLY_HANDLED__")

            if (
                previous_context
                and not is_repeat_question
                and _is_context_follow_up(user_input, previous_context)
            ):
                yield _event("status", {"text": "沿用上一份分析資料回答延伸問題..."})
                final_reply = yield from _stream_text(
                    generate_follow_up_answer_stream(
                        user_input,
                        previous_context,
                        [m.model_dump() for m in messages],
                        user_news_snippet=user_input if _looks_like_context_follow_up(user_input, previous_context) else None,
                    )
                )
                analysis_context = previous_context
            else:
                yield _event("status", {"text": "解析投資意圖與標的..."})
                try:
                    intent = _run_with_timeout(
                        "標的與意圖解析",
                        20,
                        extract_intent,
                        user_input,
                        1,
                        profile.risk_tolerance,
                        [m.model_dump() for m in messages],
                    )
                except StepTimeoutError:
                    intent = _fallback_intent(user_input, profile)
                intent["risk_tolerance"] = profile.risk_tolerance
                intent["investor_profile"] = profile.model_dump()

                if intent.get("intent_type") == "UNRELATED":
                    final_reply = "請輸入跟台股投資、金融術語或市場分析相關的問題。"
                    yield _event("token", {"text": final_reply})
                elif intent.get("intent_type") == "FINANCIAL_TERM":
                    yield _event("status", {"text": "辨識為金融術語說明..."})
                    final_reply = yield from _stream_text(generate_financial_term_answer_stream(user_input, profile.model_dump()))
                    analysis_context = None
                elif intent.get("intent_type") == "USER_NEWS":
                    yield _event("status", {"text": "分析使用者提供的新聞情緒..."})
                    analysis = analyze_user_provided_news(user_input)
                    final_reply = yield from _stream_text(
                        generate_user_news_sentiment_answer_stream(user_input, analysis, profile.model_dump())
                    )
                else:
                    rule_mentions = find_stock_mentions(user_input)
                    stock_mentions = _merge_stock_mentions(intent.get("stocks") or [], rule_mentions)
                    wants_portfolio = len(stock_mentions) > 1 and any(
                        word in user_input for word in ["手上", "現有", "持股", "健檢", "投資組合", "看看", "這些", "這幾檔"]
                    )
                    # Any 2+ stocks mentioned => compare table, even without explicit compare keywords
                    wants_compare = not wants_portfolio and len(stock_mentions) >= 2

                    if wants_portfolio or wants_compare:
                        limit = 5
                        compared_names = "、".join(
                            f"{item.get('company_name') or item.get('ticker')}（{_normalize_ticker(item.get('ticker'))}）"
                            for item in stock_mentions[:limit]
                        )
                        yield _event("token", {"text": f"目前辨識為：{compared_names}\n\n"})
                        rows = []
                        yield _event("status", {"text": "[1/4] 🧭 確認比較標的與模式..."})
                        for idx, stock_info in enumerate(stock_mentions[:limit], start=1):
                            yield _event("status", {"text": f"[2/4] 📊 蒐集第 {idx} 檔資料：{stock_info.get('company_name')}..."})
                            rows.append(_analyze_stock_for_comparison(stock_info))
                        yield _event("status", {"text": "[3/4] 🧩 彙整比較表與交叉訊號..."})
                        table_md = _format_portfolio_table(rows)
                        heading = "投資組合/持股健檢結果" if wants_portfolio else "多檔標的比較"
                        yield _event("token", {"text": f"### {heading}\n\n{table_md}\n\n"})
                        yield _event("status", {"text": "[4/4] ✍️ 生成比較分析報告..."})
                        analysis_text = ""
                        try:
                            analysis_text = yield from _stream_text(
                                generate_portfolio_analysis_stream(
                                    rows,
                                    user_input,
                                    profile.model_dump(),
                                    intent_type="PORTFOLIO" if wants_portfolio else "COMPARE",
                                )
                            )
                        except Exception:
                            analysis_text = ""
                        rec_md = _compare_rank_markdown(rows)
                        analysis_context = _build_compare_analysis_context(
                            rows,
                            stock_mentions[:limit],
                            profile,
                            heading,
                            compared_names,
                            rec_md,
                        )
                        if analysis_text:
                            final_reply = f"目前辨識為：{compared_names}\n\n### {heading}\n\n{table_md}\n\n{analysis_text}\n\n{rec_md}".strip()
                        else:
                            final_reply = f"目前辨識為：{compared_names}\n\n### {heading}\n\n{table_md}\n\n{rec_md}".strip()
                    else:
                        ticker = intent.get("ticker")
                        company_name = intent.get("company_name", "未知")

                        t0 = time.time()
                        yield _event("status", {"text": f"[1/4] 📡 抓取 {company_name}({ticker}) 即時行情..."})
                        try:
                            stock_data, df_history = _run_with_timeout("股價資料抓取", 20, fetch_realtime_stock_data, ticker)
                        except StepTimeoutError:
                            stock_data, df_history = None, None
                        if stock_data and stock_data.get("resolved_ticker"):
                            ticker = stock_data["resolved_ticker"]
                        company_name = resolve_tw_company_name(str(ticker or ""), str(company_name or ""))
                        resolved_label = f"目前辨識為：{company_name}（{_normalize_ticker(ticker)}）"
                        yield _event("token", {"text": f"{resolved_label}\n\n"})
                        t1 = time.time()

                        if _stock_lookup_failed(ticker, stock_data, df_history):
                            normalized = _normalize_ticker(ticker) or str(ticker or "").strip() or "未知"
                            if company_name and company_name not in ["未知", normalized]:
                                final_reply = f"{resolved_label}\n\n有標的但資料不足：{company_name}（{normalized}）目前無法取得足夠歷史股價資料，暫時不能進行量化分析。"
                            else:
                                final_reply = f"查無標的：找不到 {normalized} 對應的台股標的，請確認股票代號或公司名稱後再試。"
                            yield _event("token", {"text": final_reply})
                            assistant_message = ChatMessage(role="assistant", content=final_reply)
                            messages.append(assistant_message)
                            _save_state(state)
                            yield _event("done", {"message": assistant_message.model_dump()})
                            return

                        yield _event("status", {"text": "[2/4] 📰 掃描近期新聞 & 情緒分析..."})
                        try:
                            news_data = _run_with_timeout("新聞抓取", 25, fetch_stock_or_macro_sentiment, ticker, company_name, days=5)
                        except StepTimeoutError:
                            news_data = _fallback_news_data("新聞抓取失敗：新聞來源回應逾時，情緒分數先以 0 處理，量化分析仍會繼續。")
                        except Exception:
                            # Never abort the whole stream for news failures.
                            news_data = _fallback_news_data("新聞抓取失敗：本地新聞來源暫時無法取得，情緒分數先以 0 處理，量化分析仍會繼續。")
                        if news_data.get("news_count_status") == "fetch_failed":
                            yield _event("token", {"text": "新聞抓取失敗：目前無法取得近期新聞，情緒分數先以 0 處理，量化分析仍會繼續。\n\n"})
                        t2 = time.time()

                        yield _event("status", {"text": "[3/4] 🤖 量化模型運算中..."})
                        try:
                            quant_data = _run_with_timeout("量化模型", 35, run_quant_model, ticker, df_history, intent)
                        except StepTimeoutError:
                            quant_data = _fallback_quant_data("量化模型回應逾時")
                        except Exception:
                            quant_data = _fallback_quant_data("本地量化模型載入失敗")
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
                            analysis_context = dict(dashboard_payload)
                            analysis_context["mode"] = "stock"
                            yield _event("dashboard", {"dashboard_data": dashboard_payload})

                        yield _event("status", {"text": "[4/4] ✍️ 生成分析報告..."})
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
                        prefix = resolved_label
                        if news_data.get("news_count_status") == "fetch_failed":
                            prefix += "\n\n新聞抓取失敗：目前無法取得近期新聞，情緒分數先以 0 處理，量化分析仍會繼續。"
                        final_reply = f"{prefix}\n\n{final_reply}"

        assistant_message = ChatMessage(
            role="assistant",
            content=final_reply,
            dashboard_data=dashboard_payload,
            analysis_context=analysis_context,
        )
        messages.append(assistant_message)
        _save_state(state)
        yield _event("done", {"message": assistant_message.model_dump()})
    except Exception as exc:
        if str(exc) == "__EARLY_HANDLED__":
            assistant_message = ChatMessage(
                role="assistant",
                content=final_reply,
                dashboard_data=dashboard_payload,
                analysis_context=analysis_context,
            )
            messages.append(assistant_message)
            _save_state(state)
            yield _event("done", {"message": assistant_message.model_dump()})
            return
        error_message = _friendly_error_message(exc)
        
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


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, user_id: str = Query(...)) -> Dict[str, Any]:
    state = _load_state(_safe_user_id(user_id))
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del state.sessions[session_id]

    if not state.sessions:
        state.session_counter += 1
        new_session_id = f"session_{state.session_counter}"
        state.sessions[new_session_id] = [ChatMessage(role="assistant", content="您好！這是一個新的分析對話，請問今天想了解哪一檔股票？")]
        state.current_session = new_session_id
    elif state.current_session == session_id:
        state.current_session = next(reversed(state.sessions.keys()))

    _save_state(state)
    return {
        "current_session": state.current_session,
        "sessions": [
            SessionSummary(session_id=sid, title=_session_title(messages), messages=messages).model_dump()
            for sid, messages in state.sessions.items()
        ],
    }


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
    if not x_groq_api_key or not x_groq_api_key.strip():
        raise HTTPException(
            status_code=401,
            detail="缺少 API Key。請先在前端輸入您的 API Key 後再送出訊息。",
        )
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
