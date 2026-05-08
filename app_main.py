import streamlit as st
import time
import json
from pathlib import Path
import pandas as pd
import re
import uuid
import hashlib
import importlib

st.set_page_config(
    page_title="台股市場情緒量化分析與投資建議系統",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.markdown("### ⚙️ 系統設定")
    user_api_key = st.text_input("輸入您的 Groq API Key", type="password", placeholder="gsk_...", help="系統不會儲存您的金鑰，關閉網頁後即失效。")
    if user_api_key:
        st.session_state['user_api_key'] = user_api_key
        st.success("API Key 已暫存！")
    st.divider()

st.markdown("""
<style>
:root {
    --page-max-width: 1280px;
    --page-gutter: 2.5rem;
    --surface-bg: #ffffff;
    --surface-border: #e2e8f0;
    --surface-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    --composer-shadow: 0 -18px 42px rgba(15, 23, 42, 0.12);
    --accent: #0f766e;
    --accent-dark: #115e59;
}

/* 放大全域內文與一般按鈕字體 */
div[data-testid="stMarkdownContainer"] p, 
div[data-testid="stMarkdownContainer"] li {
    font-size: 18px !important;
    line-height: 1.7 !important;
}

button p, button span, div[data-testid="stSidebar"] button p {
    font-size: 18px !important;
}

/* 隱藏不需要的頂部選單、Deploy 按鈕與 footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
div[data-testid="stDecoration"] {visibility: hidden;}
[data-testid="stHeader"] {background-color: transparent;}
.stAppDeployButton {display: none !important;}

/* 確保側邊欄收合後仍可看到展開按鈕 */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 1000;
}

/* 側邊欄整體背景色與按鈕美化設計 */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #eef4f7 100%);
}

/* 統一側邊欄中所有按鈕 (Secondary / 基本按鈕) 為輕量化卡片風格 */
[data-testid="stSidebar"] button {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    color: #334155;
    text-align: left !important;
    justify-content: flex-start;
    padding: 10px 15px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    transition: all 0.2s ease-in-out;
}
[data-testid="stSidebar"] button:hover {
    background-color: #f8fafc;
    border-color: #cbd5e1;
    color: #0f172a;
    transform: translateY(-1px);
    box-shadow: 0 3px 6px rgba(0,0,0,0.05);
}

/* 整個應用的底色微調為淺灰，以突顯白色卡片 */
.stApp {
    background-color: #f3f6f8;
}

/* 主內容區重新建立穩定版面 */
.main .block-container {
    max-width: var(--page-max-width) !important;
    margin: 0 auto !important;
    min-height: 100vh;
    padding-top: 2.5rem !important;
    padding-right: var(--page-gutter) !important;
    padding-left: var(--page-gutter) !important;
    padding-bottom: 320px !important;
}

.main h1 {
    font-size: clamp(2.4rem, 4vw, 3.3rem) !important;
    line-height: 1.15 !important;
    margin-bottom: 2rem !important;
    letter-spacing: -0.03em;
}

.main h3 {
    margin-top: 0.25rem !important;
    margin-bottom: 0.9rem !important;
}

.main h4 {
    margin-top: 0.2rem !important;
    margin-bottom: 0.9rem !important;
}

/* 投資風險設定卡片 */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important;
    border: 1px solid var(--surface-border) !important;
    background-color: var(--surface-bg) !important;
    box-shadow: var(--surface-shadow) !important;
}

/* 對話區塊卡片化 */
[data-testid="stChatMessage"] {
    background-color: var(--surface-bg);
    border: 1px solid var(--surface-border);
    border-radius: 18px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--surface-shadow);
}

[data-testid="stChatMessageContent"] {
    width: 100% !important;
}

/* 標籤頁卡片化 (Dashboard內容) */
[data-testid="stTabs"] {
    background-color: var(--surface-bg);
    border: 1px solid var(--surface-border);
    border-radius: 18px;
    padding: 1.5rem;
    box-shadow: var(--surface-shadow);
    margin-top: 1rem;
}

/* 針對輸入表單移除多餘的背景與邊框，因為已經有外層的白底了 */
div[data-testid="stForm"] {
    background-color: transparent !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
    margin-bottom: 0 !important;
}

div[data-testid="stForm"] form {
    background-color: transparent !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
    margin-bottom: 0px !important;
}

div[data-testid="stForm"] form > div:first-child {
    margin: 0 auto !important;
}

/* 強制讓右側送出按鈕與輸入框高度一致並滿版 */
div[data-testid="stForm"] button {
    height: 60px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    background-color: var(--accent) !important;
    color: white !important;
    border-radius: 8px !important;
    margin-top: 0px !important;
    border: none !important;
    box-shadow: 0 10px 20px rgba(15, 118, 110, 0.2) !important;
}

div[data-testid="stForm"] button:hover {
    background-color: var(--accent-dark) !important;
    border: none !important;
}

div[data-testid="stForm"] textarea {
    min-height: 60px !important;
    background-color: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
    font-size: 17px !important;
    line-height: 1.55 !important;
}

div[data-testid="stForm"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.14) !important;
    outline: none !important;
}

div[data-testid="stForm"] [data-testid="stTextAreaRootElement"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.14) !important;
}

div[data-testid="InputInstructions"] {
    display: none !important;
}

/* 建議問句區塊在底部固定列中的排列 */
.follow-up-label {
    color: #0f172a !important;
    font-size: 1rem !important;
    font-weight: 800 !important;
    margin-bottom: 0.35rem !important;
}

.stBottom {
    background: linear-gradient(180deg, rgba(243, 246, 248, 0) 0%, rgba(243, 246, 248, 0.94) 22%, #f3f6f8 100%) !important;
    border-top: 1px solid rgba(148, 163, 184, 0.25) !important;
}

[data-testid="stBottomBlockContainer"] {
    padding: 0.85rem var(--page-gutter) 0.95rem var(--page-gutter) !important;
}

.st-key-bottom_composer_panel,
.st-key-bottom_composer_panel_fallback,
.st-key-bottom-composer-panel,
.st-key-bottom-composer-panel-fallback {
    width: min(var(--page-max-width), calc(100vw - calc(var(--page-gutter) * 2))) !important;
    margin: 0 auto 0.8rem auto !important;
    padding: 1rem 1.15rem 1.1rem 1.15rem !important;
    border: 1px solid rgba(148, 163, 184, 0.45) !important;
    border-radius: 16px !important;
    background: rgba(255, 255, 255, 0.96) !important;
    box-shadow: var(--composer-shadow) !important;
    backdrop-filter: blur(18px) saturate(145%) !important;
}

.st-key-bottom_composer_panel [data-testid="stHorizontalBlock"],
.st-key-bottom_composer_panel_fallback [data-testid="stHorizontalBlock"],
.st-key-bottom-composer-panel [data-testid="stHorizontalBlock"],
.st-key-bottom-composer-panel-fallback [data-testid="stHorizontalBlock"] {
    gap: 0.8rem !important;
}

.st-key-bottom_composer_panel [data-testid="stButton"] button,
.st-key-bottom_composer_panel_fallback [data-testid="stButton"] button,
.st-key-bottom-composer-panel [data-testid="stButton"] button,
.st-key-bottom-composer-panel-fallback [data-testid="stButton"] button {
    min-height: 42px !important;
    white-space: normal !important;
    border-radius: 999px !important;
    border: 1px solid #cbd5e1 !important;
    background: #f8fafc !important;
    color: #334155 !important;
    box-shadow: none !important;
    font-size: 15px !important;
    font-weight: 650 !important;
    line-height: 1.35 !important;
}

.st-key-bottom_composer_panel [data-testid="stButton"] button:hover,
.st-key-bottom_composer_panel_fallback [data-testid="stButton"] button:hover,
.st-key-bottom-composer-panel [data-testid="stButton"] button:hover,
.st-key-bottom-composer-panel-fallback [data-testid="stButton"] button:hover {
    border-color: var(--accent) !important;
    background: #ecfdf5 !important;
    color: var(--accent-dark) !important;
}

.st-key-bottom_composer_panel [data-testid="stForm"] [data-testid="stButton"] button,
.st-key-bottom_composer_panel_fallback [data-testid="stForm"] [data-testid="stButton"] button,
.st-key-bottom-composer-panel [data-testid="stForm"] [data-testid="stButton"] button,
.st-key-bottom-composer-panel-fallback [data-testid="stForm"] [data-testid="stButton"] button {
    border-radius: 10px !important;
    background-color: var(--accent) !important;
    color: #ffffff !important;
    font-size: 17px !important;
    font-weight: 800 !important;
    box-shadow: 0 10px 20px rgba(15, 118, 110, 0.2) !important;
}

.st-key-bottom_composer_panel [data-testid="stForm"] [data-testid="stButton"] button:hover,
.st-key-bottom_composer_panel_fallback [data-testid="stForm"] [data-testid="stButton"] button:hover,
.st-key-bottom-composer-panel [data-testid="stForm"] [data-testid="stButton"] button:hover,
.st-key-bottom-composer-panel-fallback [data-testid="stForm"] [data-testid="stButton"] button:hover {
    background-color: var(--accent-dark) !important;
    color: #ffffff !important;
}

@media (max-width: 1200px) {
    :root {
        --page-gutter: 1.5rem;
    }
}

@media (max-width: 768px) {
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 420px !important;
    }

    :root {
        --page-gutter: 1rem;
    }

    [data-testid="stBottomBlockContainer"] {
        padding: 0.65rem var(--page-gutter) 0.75rem var(--page-gutter) !important;
    }

    .st-key-bottom_composer_panel,
    .st-key-bottom_composer_panel_fallback,
    .st-key-bottom-composer-panel,
    .st-key-bottom-composer-panel-fallback {
        padding: 0.85rem !important;
        margin-bottom: 0.55rem !important;
        border-radius: 14px !important;
    }

    .st-key-bottom_composer_panel [data-testid="stButton"] button,
    .st-key-bottom_composer_panel_fallback [data-testid="stButton"] button,
    .st-key-bottom-composer-panel [data-testid="stButton"] button,
    .st-key-bottom-composer-panel-fallback [data-testid="stButton"] button {
        font-size: 14px !important;
    }
}
</style>
""", unsafe_allow_html=True)

import data_fetch as data_fetch_module

data_fetch_module = importlib.reload(data_fetch_module)
analyze_user_provided_news = data_fetch_module.analyze_user_provided_news
extract_intent = data_fetch_module.extract_intent
fetch_realtime_stock_data = data_fetch_module.fetch_realtime_stock_data
fetch_stock_or_macro_sentiment = data_fetch_module.fetch_stock_or_macro_sentiment
find_stock_mentions = data_fetch_module.find_stock_mentions
generate_financial_term_answer_stream = data_fetch_module.generate_financial_term_answer_stream
generate_follow_up_answer_stream = data_fetch_module.generate_follow_up_answer_stream
generate_investment_advice_stream = data_fetch_module.generate_investment_advice_stream
generate_user_news_sentiment_answer_stream = data_fetch_module.generate_user_news_sentiment_answer_stream
run_quant_model = data_fetch_module.run_quant_model
from ui_components import render_dashboard

# ==========================================
# 狀態管理初始化
# ==========================================
APP_DIR = Path(__file__).resolve().parent
USER_HISTORY_DIR = APP_DIR / "data" / "user_histories"

DEFAULT_ASSISTANT_MESSAGE = "您好！我是您的台股投資顧問，您可以在底下直接輸入標的（如：「台積電走勢如何？」），系統會自動為您分析。"

def get_or_create_user_id():
    user_id = st.query_params.get("user_id")
    if isinstance(user_id, list):
        user_id = user_id[0] if user_id else None
    if not user_id or not re.fullmatch(r"[a-zA-Z0-9_-]{8,64}", str(user_id)):
        user_id = uuid.uuid4().hex
        st.query_params["user_id"] = user_id
    return str(user_id)

def get_chat_store_path(user_id):
    safe_id = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]
    return USER_HISTORY_DIR / f"{safe_id}.json"

USER_ID = get_or_create_user_id()
CHAT_STORE_PATH = get_chat_store_path(USER_ID)

def _serialize_dashboard_data(dashboard_data):
    if not dashboard_data:
        return None

    serialized = dict(dashboard_data)
    df_history = serialized.get("df_history")
    if isinstance(df_history, pd.DataFrame):
        df_to_store = df_history.copy()
        df_to_store.index = df_to_store.index.astype(str)
        serialized["df_history"] = {
            "__type__": "dataframe",
            "index": df_to_store.index.tolist(),
            "records": df_to_store.reset_index(drop=True).to_dict(orient="records"),
        }
    return serialized

def _deserialize_dashboard_data(dashboard_data):
    if not dashboard_data:
        return None

    deserialized = dict(dashboard_data)
    df_payload = deserialized.get("df_history")
    if isinstance(df_payload, dict) and df_payload.get("__type__") == "dataframe":
        df_history = pd.DataFrame(df_payload.get("records", []))
        if not df_history.empty:
            df_history.index = pd.to_datetime(df_payload.get("index", []), errors="coerce")
        deserialized["df_history"] = df_history
    return deserialized

def _serialize_sessions(sessions):
    payload = {}
    for session_id, messages in sessions.items():
        payload[session_id] = []
        for msg in messages:
            stored_msg = dict(msg)
            if "dashboard_data" in stored_msg:
                stored_msg["dashboard_data"] = _serialize_dashboard_data(stored_msg["dashboard_data"])
            payload[session_id].append(stored_msg)
    return payload

def _deserialize_sessions(sessions):
    restored = {}
    for session_id, messages in sessions.items():
        restored[session_id] = []
        for msg in messages:
            restored_msg = dict(msg)
            if "dashboard_data" in restored_msg:
                restored_msg["dashboard_data"] = _deserialize_dashboard_data(restored_msg["dashboard_data"])
            restored[session_id].append(restored_msg)
    return restored

def load_app_state():
    if not CHAT_STORE_PATH.exists():
        return None
    try:
        with CHAT_STORE_PATH.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not payload.get("sessions"):
            return None
        return {
            "sessions": _deserialize_sessions(payload["sessions"]),
            "current_session": payload.get("current_session", "session_1"),
            "session_counter": int(payload.get("session_counter", 1)),
            "investor_profile": payload.get("investor_profile", {}),
        }
    except Exception as e:
        st.warning(f"歷史紀錄讀取失敗，已改用新的對話狀態：{e}")
        return None

def save_app_state():
    try:
        CHAT_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sessions": _serialize_sessions(st.session_state.sessions),
            "current_session": st.session_state.current_session,
            "session_counter": st.session_state.session_counter,
            "investor_profile": st.session_state.investor_profile,
        }
        with CHAT_STORE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.toast(f"歷史紀錄儲存失敗：{e}", icon="⚠️")

def build_investor_profile():
    style = st.session_state.get("investment_style", "穩健")
    risk_tolerance = risk_map.get(style, st.session_state.get("risk_tolerance", "medium"))
    st.session_state.risk_tolerance = risk_tolerance
    return {
        "style": style,
        "risk_tolerance": risk_tolerance,
        "max_loss_pct": st.session_state.get("max_loss_pct", 10),
    }

def update_investor_profile():
    st.session_state.investor_profile = build_investor_profile()
    save_app_state()

def format_current_profile_summary():
    return (
        f"{st.session_state.investment_style}｜"
        f"最大虧損 {st.session_state.max_loss_pct}%"
    )

def get_latest_analysis_context(messages):
    last_user = ""
    last_dashboard = None
    for msg in messages:
        if msg.get("role") == "user":
            last_user = msg.get("content", "")
        if msg.get("dashboard_data"):
            last_dashboard = msg["dashboard_data"]
    return last_user, last_dashboard

def normalize_ticker(value):
    if not value:
        return ""
    return str(value).upper().replace(".TW", "").replace(".TWO", "").strip()

def is_same_stock_follow_up(question, dashboard_data):
    if not dashboard_data:
        return False

    context_ticker = normalize_ticker(dashboard_data.get("ticker"))
    ticker_mentions = re.findall(r'(?<!\d)(\d{4})(?!\d)', question)
    if ticker_mentions:
        return context_ticker in ticker_mentions

    return True

def make_follow_up_questions(user_question, dashboard_data):
    if not dashboard_data:
        # 處理沒有單一 dashboard_data 的情況（例如多檔股票、新聞分析）
        if any(keyword in user_question for keyword in ["比較", "哪個", "選", "vs"]):
            return [
                "如果資金有限，這幾檔股票應該優先佈局哪一檔？",
                "這幾檔標的中，哪一檔的下檔風險（回撤）相對可控？",
                "這幾檔股票的近期法人籌碼動向如何？",
            ]
        elif any(keyword in user_question for keyword in ["新聞", "網址", "http"]):
            return [
                "這則新聞對股價的影響會是短線波動還是長線趨勢？",
                "除了這則新聞，近期還有哪些潛在的未爆彈或利多需要注意？",
                "如果因為這則新聞而進場，停損點應該設在哪？",
            ]
        else:
            return [
                "目前這幾檔股票，哪一檔的量化訊號最明確？",
                "如果大盤接下來轉弱，這些持股該怎麼調整比例？",
                "可以幫我詳細評估其中最危險的那檔股票嗎？",
            ]

    ticker = dashboard_data.get("ticker", "")
    company_name = dashboard_data.get("company_name") or ticker
    company = f"{company_name}（{ticker}）" if ticker and company_name != ticker else ticker
    stock_data = dashboard_data.get("stock_data", {})
    quant_data = dashboard_data.get("quant_data", {})
    signal = quant_data.get("signal", "觀望")
    trend = stock_data.get("trend_summary", "目前趨勢")

    question_sets = []
    if any(keyword in user_question for keyword in ["風險", "最大風險", "跌", "停損"]):
        question_sets = [
            f"{company} 如果跌破月線，停損要設在哪裡？",
            f"{company} 目前最需要觀察哪個風險訊號？",
            f"{company} 適合分批買進還是先等回檔？",
            f"{company} 如果新聞轉空，我應該怎麼調整部位？",
        ]
    elif any(keyword in user_question for keyword in ["續抱", "減碼", "停利"]):
        question_sets = [
            f"{company} 目前續抱需要看哪三個條件？",
            f"{company} 如果要分批減碼，該看哪些訊號？",
            f"{company} 如果想提高勝率，應該等什麼確認訊號？",
            f"{company} 我的停利與停損可以怎麼搭配？",
        ]
    elif any(keyword in user_question for keyword in ["進場", "買", "買入", "加碼"]):
        question_sets = [
            f"{company} 現在適合一次買還是分批買？",
            f"{company} 進場前需要等哪些確認訊號？",
            f"{company} 如果量化買入但情緒偏弱怎麼辦？",
            f"{company} 這個價位的風險報酬比合理嗎？",
        ]
    else:
        question_sets = [
            f"{company} 的 {signal} 訊號可以怎麼設定進出場？",
            f"{company} 目前適合觀望、分批還是等待突破？",
            f"{company} 在{trend}的情況下，接下來最大風險是什麼？",
            f"{company} 的建議可信度主要受哪些資料影響？",
        ]

    return question_sets

def get_asked_questions(messages):
    return {
        msg.get("content", "").strip()
        for msg in messages
        if msg.get("role") == "user" and msg.get("content")
    }

def has_user_asked(messages):
    return any(msg.get("role") == "user" and msg.get("content", "").strip() for msg in messages)

def select_new_follow_up_questions(messages, user_question, dashboard_data):
    asked_questions = get_asked_questions(messages)
    
    # 根據上一則使用者的對話動態抓取邏輯
    last_user_msg = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), user_question)
    
    candidates = make_follow_up_questions(last_user_msg, dashboard_data)
    fresh_questions = [question for question in candidates if question.strip() not in asked_questions]
    
    if not dashboard_data:
        return fresh_questions[:3]
    
    ticker = dashboard_data.get("ticker", "")
    company_name = dashboard_data.get("company_name") or ticker
    company = f"{company_name}（{ticker}）" if ticker and company_name != ticker else ticker
    
    is_price_q = any(q in last_user_msg for q in ["價", "買", "賣", "支撐", "壓力", "會不會跌", "突破", "進場"])
    is_news_q = any(q in last_user_msg for q in ["新聞", "情緒", "消息", "利多", "利空", "大增", "財報"])
    
    if is_price_q:
        fallback_questions = [
            f"{company} 如果接下來價格跌破支撐，該如何應對？",
            f"{company} 近期最大的回撤風險可以如何控管？",
            f"大盤若轉弱，{company} 還有抗跌的空間嗎？",
        ]
    elif is_news_q:
        fallback_questions = [
            f"這些新聞情緒對 {company} 的影響大約會持續多久？",
            f"{company} 的近期籌碼面有受到這波情緒帶動嗎？",
            f"若後續還有壞消息，{company} 還值得留嗎？",
        ]
    else:
        fallback_questions = [
            f"{company} 接下來我應該觀察哪些關鍵指標？",
            f"{company} 目前的建議可信度為什麼是這個等級？",
            f"{company} 這樣的量化訊號建議怎樣分批佈局？",
        ]

    for question in fallback_questions:
        if question.strip() not in asked_questions and question not in fresh_questions:
            fresh_questions.append(question)
    return fresh_questions[:3]

def render_follow_up_buttons(messages):
    if not has_user_asked(messages):
        return

    user_question, dashboard_data = get_latest_analysis_context(messages)
    questions = select_new_follow_up_questions(messages, user_question, dashboard_data)
    if not questions:
        return

    st.markdown('<div class="follow-up-label">💡 你可能會想繼續問：</div>', unsafe_allow_html=True)
    
    if questions:
        cols = st.columns(len(questions))
        for idx, q in enumerate(questions):
            with cols[idx]:
                # 這裡的 st.button 會套用剛剛設定的全域放大 18px CSS
                if st.button(q, key=f"follow_up_{st.session_state.current_session}_{len(messages)}_{idx}", use_container_width=True):
                    if not st.session_state.get("next_prompt"):
                        set_follow_up_prompt(q, dashboard_data)
                        st.rerun()




def analyze_stock_for_comparison(stock_info):
    ticker = stock_info.get("ticker")
    company_name = stock_info.get("company_name") or ticker
    stock_data, df_history = fetch_realtime_stock_data(ticker)
    resolved_ticker = stock_data.get("resolved_ticker") if stock_data else None
    display_ticker = resolved_ticker or normalize_ticker(ticker) or ticker
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
        "_predicted_return": quant_data.get("predicted_return", 0),
        "_sentiment_score": news_data.get("sentiment_score", 0),
    }

def normalize_compare_stock_mentions(stock_mentions):
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

def format_portfolio_table(rows):
    display_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    columns = list(display_rows[0].keys()) if display_rows else []
    table_lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in display_rows:
        table_lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(table_lines)

def render_investor_profile_panel():
    st.markdown("### 投資人風險設定")
    with st.container(border=True):
        st.caption(f"目前設定：{format_current_profile_summary()}")

        style_col, loss_col = st.columns(2)
        with style_col:
            st.markdown("#### 投資風格")
            selected_style = st.radio(
                "投資風格",
                style_options,
                horizontal=True,
                key="investment_style",
                on_change=update_investor_profile,
                label_visibility="collapsed",
            )
            st.session_state.risk_tolerance = risk_map.get(selected_style, "medium")
        with loss_col:
            st.markdown("#### 最大可接受虧損")
            st.select_slider(
                "最大可接受虧損",
                options=[5, 10, 15, 20, 30],
                format_func=lambda value: f"{value}%",
                key="max_loss_pct",
                on_change=update_investor_profile,
                label_visibility="collapsed",
            )

        update_investor_profile()

saved_state = load_app_state()
if "sessions" not in st.session_state:
    if saved_state:
        st.session_state.sessions = saved_state["sessions"]
    else:
        st.session_state.sessions = {
            "session_1": [{"role": "assistant", "content": DEFAULT_ASSISTANT_MESSAGE}]
        }

if "current_session" not in st.session_state:
    st.session_state.current_session = saved_state["current_session"] if saved_state else "session_1"
    if st.session_state.current_session not in st.session_state.sessions:
        st.session_state.current_session = next(iter(st.session_state.sessions.keys()))
if "session_counter" not in st.session_state:
    st.session_state.session_counter = saved_state["session_counter"] if saved_state else 1
if "next_prompt" not in st.session_state:
    st.session_state.next_prompt = None
if "pending_follow_up_dashboard" not in st.session_state:
    st.session_state.pending_follow_up_dashboard = None
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = None
if "investor_profile" not in st.session_state:
    st.session_state.investor_profile = saved_state["investor_profile"] if saved_state else {
        "style": "穩健",
        "risk_tolerance": "medium",
        "max_loss_pct": 10,
    }

profile = st.session_state.investor_profile
style_options = ["保守", "穩健", "積極"]
risk_map = {"保守": "low", "穩健": "medium", "積極": "high"}
reverse_risk_map = {"low": "保守", "medium": "穩健", "high": "積極"}
if "investment_style" not in st.session_state:
    st.session_state.investment_style = profile.get("style") or reverse_risk_map.get(profile.get("risk_tolerance"), "穩健")
if "risk_tolerance" not in st.session_state:
    st.session_state.risk_tolerance = profile.get("risk_tolerance", risk_map.get(st.session_state.investment_style, "medium"))
if "max_loss_pct" not in st.session_state:
    st.session_state.max_loss_pct = int(profile.get("max_loss_pct", 10))
def set_quick_prompt(text):
    st.session_state.next_prompt = text
    st.session_state.pending_follow_up_dashboard = None

def set_follow_up_prompt(text, dashboard_data):
    st.session_state.next_prompt = text
    st.session_state.pending_follow_up_dashboard = dashboard_data

def create_new_chat():
    st.session_state.session_counter += 1
    new_session_id = f"session_{st.session_state.session_counter}"
    st.session_state.sessions[new_session_id] = [{"role": "assistant", "content": "您好！這是一個新的分析對話 \n請問今天想了解哪一檔股票？"}]
    st.session_state.current_session = new_session_id
    st.session_state.current_ticker = None 
    save_app_state()

def switch_chat(session_id):
    st.session_state.current_session = session_id
    st.session_state.current_ticker = None
    save_app_state()

def get_session_title(session_messages):
    for msg in session_messages:
        if msg["role"] == "user":
            return msg["content"][:15] + ("..." if len(msg["content"]) > 15 else "")
    return "新對話"

with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding-bottom: 10px;'>
            <h2 style='margin-bottom: 5px;'>台股投資顧問</h2>
            
    """, unsafe_allow_html=True)
    st.button("➕ 開啟新對話", key="new_chat_btn", on_click=create_new_chat, use_container_width=True, type="primary")
    st.divider()
    st.markdown("### 歷史紀錄")
    for session_id in reversed(list(st.session_state.sessions.keys())):
        session_title = get_session_title(st.session_state.sessions[session_id])
        btn_label = f"📌 {session_title}" if session_id == st.session_state.current_session else session_title
        st.button(btn_label, key=f"btn_{session_id}", on_click=switch_chat, args=(session_id,), use_container_width=True)

# ==========================================
# Streamlit 介面渲染
# ==========================================
st.title("📊 台股市場情緒量化分析與投資建議系統")

render_investor_profile_panel()

current_messages = st.session_state.sessions[st.session_state.current_session]

for msg in current_messages:
    with st.chat_message(msg["role"]):
        if "dashboard_data" in msg:
            render_dashboard(msg["dashboard_data"])
        st.markdown(msg["content"])

user_input = None
if st.session_state.next_prompt:
    user_input = st.session_state.next_prompt
    follow_up_dashboard = st.session_state.pending_follow_up_dashboard
    st.session_state.next_prompt = None
    st.session_state.pending_follow_up_dashboard = None

if user_input:
    current_messages.append({"role": "user", "content": user_input})
    save_app_state()
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if follow_up_dashboard and is_same_stock_follow_up(user_input, follow_up_dashboard):
            stream_gen = generate_follow_up_answer_stream(user_input, follow_up_dashboard, current_messages)
            final_reply = st.write_stream(stream_gen)
            current_messages.append({"role": "assistant", "content": final_reply})
            save_app_state()
            st.rerun()

        with st.status("啟動投資顧問分析引擎...", expanded=True) as status:
            import config
            if not st.session_state.get("user_api_key") and not config._get_secret("GROQ_API_KEY"):
                status.update(label="⚠️ 缺乏 API Key", state="error", expanded=False)
                st.error("請先於左側欄輸入您的 Groq API Key 以啟用 AI 分析功能！")
                st.stop()
                
            t_overall_start = time.time()
            progress_bar = st.progress(0, text="初始化中...")
            
            st.write("🔍 正在解析您的投資意圖與標的...")
            progress_bar.progress(10, text="解析語意意圖...")

            t_step_start = time.time()
            investor_profile = build_investor_profile()
            intent = extract_intent(user_input, 1, investor_profile["risk_tolerance"], current_messages)
            intent["risk_tolerance"] = investor_profile["risk_tolerance"]
            intent["investor_profile"] = investor_profile
            t_step_end = time.time()
            st.write(f"⏱️ 意圖解析耗時: {t_step_end - t_step_start:.2f} 秒")
            
            if intent.get("intent_type") == "UNRELATED":
                status.update(label="⚠️ 偵測到非投資相關對話", state="error", expanded=False)
                reply_msg = "不好意思，請輸入與投資有關的問題喔！我是專注於台股量化與情緒分析的 AI 投資顧問。"
                st.markdown(reply_msg)
                current_messages.append({"role": "assistant", "content": reply_msg})
                save_app_state()
                st.rerun()

            if intent.get("intent_type") == "FINANCIAL_TERM":
                status.update(label="✅ 已辨識為金融術語說明", state="complete", expanded=False)
                stream_gen = generate_financial_term_answer_stream(user_input, investor_profile)
                final_reply = st.write_stream(stream_gen)
                current_messages.append({"role": "assistant", "content": final_reply})
                save_app_state()
                st.rerun()

            if intent.get("intent_type") == "USER_NEWS":
                status.update(label="📰 已辨識為使用者提供新聞，正在判讀情緒", state="running", expanded=True)
                progress_bar.progress(55, text="分析使用者提供的新聞情緒...")
                analysis = analyze_user_provided_news(user_input)
                st.success(f"✅ 新聞情緒判讀完成：{analysis.get('overall_label')}，綜合分數 {analysis.get('sentiment_score')}")
                status.update(label="✅ 新聞情緒判讀完成", state="complete", expanded=False)
                stream_gen = generate_user_news_sentiment_answer_stream(user_input, analysis, investor_profile)
                final_reply = st.write_stream(stream_gen)
                current_messages.append({"role": "assistant", "content": final_reply})
                save_app_state()
                st.rerun()

            stock_mentions = normalize_compare_stock_mentions(intent.get("stocks") or find_stock_mentions(user_input))
            wants_portfolio = len(stock_mentions) > 1 and any(word in user_input for word in ["手上", "現有", "持股", "健檢", "投資組合", "看看", "這些", "這幾檔", "分析"])
            wants_compare = not wants_portfolio and (
                intent.get("intent_type") == "STOCK_COMPARE" or (
                    len(stock_mentions) >= 2 and any(word in user_input for word in ["比較", "哪個", "誰", "vs", "VS", "差異", "選"])
                )
            )
            
            if wants_portfolio:
                status.update(label="💼 正在執行投資組合分析", state="running", expanded=True)
                rows = []
                for idx, stock_info in enumerate(stock_mentions[:5], start=1):
                    st.write(f"🔎 正在分析第 {idx} 檔：{stock_info.get('company_name')} ({stock_info.get('ticker')})")
                    progress_bar.progress(10 + idx * 15, text=f"健檢 {stock_info.get('company_name')}...")
                    rows.append(analyze_stock_for_comparison(stock_info))
                
                status.update(label="✅ 數據彙整完成，AI 正在產生深入分析...", state="complete", expanded=False)
                
                table_md = format_portfolio_table(rows)
                
                from data_fetch import generate_portfolio_analysis_stream
                with st.chat_message("assistant"):
                    st.markdown(f"### 💼 投資組合/持股健檢結果\n\n{table_md}\n\n")
                    response_stream = generate_portfolio_analysis_stream(rows, user_input, investor_profile, intent_type="PORTFOLIO")
                    final_analysis = st.write_stream(response_stream)
                
                final_reply = f"### 💼 投資組合/持股健檢結果\n\n{table_md}\n\n{final_analysis}"
                current_messages.append({"role": "assistant", "content": final_reply})
                save_app_state()
                st.rerun()

            elif wants_compare and len(stock_mentions) >= 2:
                status.update(label="📊 正在比較兩檔股票", state="running", expanded=True)
                rows = []
                for idx, stock_info in enumerate(stock_mentions[:2], start=1):
                    st.write(f"🔎 正在分析第 {idx} 檔：{stock_info.get('company_name')} ({stock_info.get('ticker')})")
                    progress_bar.progress(25 + idx * 30, text=f"分析 {stock_info.get('company_name')}...")
                    rows.append(analyze_stock_for_comparison(stock_info))
                
                status.update(label="✅ 數據比較完成，AI 正在產生深入分析...", state="complete", expanded=False)
                
                table_md = format_portfolio_table(rows)
                
                from data_fetch import generate_portfolio_analysis_stream
                with st.chat_message("assistant"):
                    st.markdown(f"### 📊 兩檔標的比較\n\n{table_md}\n\n")
                    response_stream = generate_portfolio_analysis_stream(rows, user_input, investor_profile, intent_type="COMPARE")
                    final_analysis = st.write_stream(response_stream)
                
                final_reply = f"### 📊 兩檔標的比較\n\n{table_md}\n\n{final_analysis}"
                current_messages.append({"role": "assistant", "content": final_reply})
                save_app_state()
                st.rerun()

            ticker = intent.get('ticker')
            st.info(f"🎯 **確認查詢標的**：{intent.get('company_name', '未知')} ({ticker})")
            
            st.write(f"📊 正在抓取 {intent.get('company_name', '未知')} ({ticker}) 的即時行情...")
            progress_bar.progress(35, text="取得即時行情報價...")
            
            t_step_start = time.time()
            stock_data, df_history = fetch_realtime_stock_data(ticker)
            t_step_end = time.time()
            
            if stock_data and stock_data.get("resolved_ticker"):
                ticker = stock_data["resolved_ticker"]
                
            st.write(f"⏱️ 行情抓取耗時: {t_step_end - t_step_start:.2f} 秒")
            st.success(f"✅ 成功擷取歷史行情數據，並確認目前收盤狀態。")
            
            st.write(f"📰 正在掃描過去 5 天相關的新聞與計算 FinBERT...")
            progress_bar.progress(60, text="抓取新聞並執行情緒過濾...")
            
            t_step_start = time.time()
            news_data = fetch_stock_or_macro_sentiment(ticker, intent.get('company_name', '未知'), days=5)
            t_step_end = time.time()
            
            st.write(f"⏱️ 新聞抓取與情緒分析耗時: {t_step_end - t_step_start:.2f} 秒")
            if news_data.get("news_count_status") == "none":
                st.warning(f"⚠️ 過去 5 天查無專屬新聞，將切換依賴量化與技術面數據。")
            else:
                st.success(f"✅ 成功抓取與解析 {len(news_data.get('raw_news', []))} 篇新聞情緒。")
            
            st.write("🤖 正在執行多重投資組合策略與量化預測...")
            progress_bar.progress(85, text="載入 Ensemble 模型參數並進行量化運算...")
            
            t_step_start = time.time()
            quant_data = run_quant_model(ticker, df_history, intent)
            t_step_end = time.time()
            
            st.write(f"⏱️ 量化模型預算耗時: {t_step_end - t_step_start:.2f} 秒")
            st.success(f"✅ 量化模型預測完畢，取得防禦性交易訊號！")

            elapsed = time.time() - t_overall_start
            progress_bar.progress(100, text=f"分析完成！(總耗時 {elapsed:.1f} 秒)")

        status.update(label=f"✅ 分析完成 (總運算耗時: {elapsed:.1f}s)", state="complete", expanded=True)

        dashboard_payload = None
        if ticker != '未知' and ticker != '未知標的':
            dashboard_payload = {
                "ticker": ticker,
                "company_name": intent.get('company_name', ticker),
                "stock_data": stock_data, "news_data": news_data,
                "quant_data": quant_data, "df_history": df_history.tail(30) if df_history is not None else None,
                "risk": intent.get('risk_tolerance') or 'medium',
                "investor_profile": investor_profile,
            }
            render_dashboard(dashboard_payload)
            st.session_state.current_ticker = ticker

        stream_gen = generate_investment_advice_stream(user_input, intent, quant_data, stock_data, news_data, current_messages, investor_profile)
        final_reply = st.write_stream(stream_gen)
        
        msg_dict = {"role": "assistant", "content": final_reply}
        if dashboard_payload:
            msg_dict["dashboard_data"] = dashboard_payload
        current_messages.append(msg_dict)
        save_app_state()

        st.rerun()

# --- 在介面最底下渲染輸入區塊 ---
def handle_chat_submit():
    text = st.session_state.get("chat_text_area", "").strip()
    if text:
        st.session_state.next_prompt = text
        st.session_state.chat_text_area = ""

bottom_container = st.bottom if hasattr(st, "bottom") else st.container()
composer_key = "bottom_composer_panel" if hasattr(st, "bottom") else "bottom_composer_panel_fallback"

with bottom_container:
    with st.container(key=composer_key):
        # 僅在使用者至少送出過一次問題後才顯示建議問句
        render_follow_up_buttons(current_messages)
        
        with st.form("chat_form", clear_on_submit=True, border=False):
            col1, col2 = st.columns([8, 1.35], vertical_alignment="bottom")
            with col1:
                st.text_area(
                    "輸入框", 
                    key="chat_text_area", 
                    height=60, 
                    label_visibility="collapsed",
                    placeholder="請輸入欲分析的投資標的，或貼上新聞內容"
                )
            with col2:
                st.form_submit_button("送出", on_click=handle_chat_submit, use_container_width=True)
