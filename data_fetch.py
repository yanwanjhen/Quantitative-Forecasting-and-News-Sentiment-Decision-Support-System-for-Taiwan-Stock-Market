import json
import time
import requests
import urllib.parse
import xml.etree.ElementTree as ET
import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os
import hashlib
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from copy import deepcopy

from pathlib import Path

# Add project root to sys.path to access experiments module
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import model_llm

API_CACHE_PATH = Path(_project_root) / "data" / "api_cache.json"
def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip() == "1"

USE_AI_NEWS_FILTER = _env_flag("USE_AI_NEWS_FILTER", "0")
ENABLE_QUANT_MODEL = _env_flag("ENABLE_QUANT_MODEL", "1")
ENABLE_FINBERT = _env_flag("ENABLE_FINBERT", "1")
EXTERNAL_STOCK_MAP_PATH = Path(_project_root) / "data" / "tw_stock_map.json"
MAX_SENTIMENT_NEWS = 30
_MEMOIZED_CACHE: dict[tuple, tuple[float, object]] = {}
NEWS_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml,application/xml,text/xml,application/json,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def _fetch_news_url(url, source_label, timeout=6, retries=2):
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=NEWS_HTTP_HEADERS, timeout=timeout + attempt * 2)
            if response.status_code in {403, 429}:
                last_error = f"HTTP {response.status_code}"
                if attempt < retries - 1:
                    time.sleep(0.6 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(0.6 * (attempt + 1))
    print(f"❌ {source_label} 抓取失敗: {last_error}")
    return None


def _cached_news_source(source, cache_identity, fetcher, ttl=1800):
    key_digest = hashlib.sha256(json.dumps(cache_identity, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    key = ("news_source", source, key_digest)
    now = time.time()
    cached = _MEMOIZED_CACHE.get(key)
    if cached and now - cached[0] < ttl:
        return deepcopy(cached[1])
    result = fetcher()
    if result is not None:
        _MEMOIZED_CACHE[key] = (now, deepcopy(result))
    return deepcopy(result)


def _load_quant_dependencies():
    import torch
    from experiments.config import get_config
    from experiments.feature_engineering import FeatureEngineer
    from experiments.models import RegimeEmbeddingLSTM

    return torch, get_config, FeatureEngineer, RegimeEmbeddingLSTM


def _safe_extract_keywords(text):
    try:
        from sentiment_analysis import extract_keywords

        return extract_keywords(text)
    except Exception:
        hits_pos, hits_neg = _keyword_hits(str(text or ""))
        return "、".join(hits_pos), "、".join(hits_neg)


def _keyword_sentiment_score(text):
    from sentiment_analysis import _keyword_hits

    pos_hits, neg_hits = _keyword_hits(str(text or ""))
    if not pos_hits and not neg_hits:
        return 0.0
    raw = (len(pos_hits) - len(neg_hits) * 1.45) / 2.0
    return float(np.tanh(raw))


def _safe_sentiment_score(text, target_company=None):
    if not ENABLE_FINBERT:
        return _keyword_sentiment_score(text)
    try:
        from sentiment_analysis import get_finbert_continuous_score

        return get_finbert_continuous_score(text, target_company=target_company)
    except Exception as exc:
        print(f"⚠️ FinBERT 不可用，改用關鍵字備援情緒分數：{exc}")
        return _keyword_sentiment_score(text)


def _lightweight_quant_fallback(df_history, reason="本地量化模型載入失敗，暫用價格趨勢備援"):
    if df_history is None or df_history.empty:
        return {
            "predicted_return": 0.0,
            "signal": "無法預測",
            "regime": "無資料",
            "max_dd": "N/A",
            "model_name": "lightweight_price_fallback",
            "model_count": 0,
            "error_message": reason,
        }

    df = df_history.dropna(subset=["Close"]).copy()
    if df.empty:
        return {
            "predicted_return": 0.0,
            "signal": "無法預測",
            "regime": "無資料",
            "max_dd": "N/A",
            "model_name": "lightweight_price_fallback",
            "model_count": 0,
            "error_message": reason,
        }

    close = df["Close"].astype(float)
    latest = float(close.iloc[-1])
    ma20 = float(close.tail(20).mean()) if len(close) >= 20 else latest
    ma60 = float(close.tail(60).mean()) if len(close) >= 60 else ma20
    momentum_20d = float(close.pct_change(20).iloc[-1]) if len(close) > 20 else 0.0
    avg_ret = max(min(momentum_20d / 20, 0.015), -0.015)

    past_year_close = close.tail(252)
    rolling_max_1y = past_year_close.expanding().max()
    drawdowns_1y = (past_year_close - rolling_max_1y) / rolling_max_1y
    max_dd_1y = float(drawdowns_1y.min()) if not drawdowns_1y.empty else 0.0

    if latest > ma20 > ma60 and avg_ret > 0.002:
        signal = "買入"
    elif latest < ma20 and avg_ret < -0.002:
        signal = "賣出"
    else:
        signal = "觀望"

    if max_dd_1y <= -0.25:
        regime = "趨勢轉弱 (熊市)"
    elif latest > ma20 > ma60:
        regime = "強勢偏多"
    else:
        regime = "回檔整理"

    return {
        "predicted_return": avg_ret,
        "signal": signal,
        "regime": regime,
        "max_dd": f"({max_dd_1y*100:.1f}%)",
        "model_name": "lightweight_price_fallback",
        "model_count": 0,
        "error_message": reason,
    }


def ttl_cache_data(ttl: int = 1800):
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = (func.__name__, repr(args), repr(sorted(kwargs.items())))
            now = time.time()
            cached = _MEMOIZED_CACHE.get(key)
            if cached and now - cached[0] < ttl:
                return deepcopy(cached[1])
            result = func(*args, **kwargs)
            should_cache = True
            if isinstance(result, dict) and result.get("news_count_status") == "fetch_failed":
                should_cache = False
            if should_cache:
                _MEMOIZED_CACHE[key] = (now, deepcopy(result))
            return deepcopy(result)

        return wrapper

    return decorator

MODEL_FEATURE_SETS = {
    "stock_tech2": [
        "log_return", "log_return_5d", "log_return_10d",
        "volatility_5d", "volatility_10d",
        "BB_width", "BB_pct",
        "ADX_14", "RSI_14", "TRIX_14",
    ],
}

def normalize_features_rolling_for_inference(df, feature_cols, window=252):
    normalized = df.copy()
    for col in feature_cols:
        rolling_mean = normalized[col].rolling(window=window, min_periods=20).mean()
        rolling_std = normalized[col].rolling(window=window, min_periods=20).std()
        normalized[col] = (normalized[col] - rolling_mean) / rolling_std.replace(0, np.nan)
    normalized[feature_cols] = normalized[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return normalized

COMMON_TW_STOCKS = {
    "台積電": ("2330.TW", "台積電"),
    "鴻海": ("2317.TW", "鴻海"),
    "聯發科": ("2454.TW", "聯發科"),
    "聯電": ("2303.TW", "聯電"),
    "台達電": ("2308.TW", "台達電"),
    "廣達": ("2382.TW", "廣達"),
    "緯創": ("3231.TW", "緯創"),
    "仁寶": ("2324.TW", "仁寶"),
    "長榮": ("2603.TW", "長榮"),
    "陽明": ("2609.TW", "陽明"),
    "萬海": ("2615.TW", "萬海"),
    "富邦金": ("2881.TW", "富邦金"),
    "國泰金": ("2882.TW", "國泰金"),
    "玉山金": ("2884.TW", "玉山金"),
    "中華電": ("2412.TW", "中華電"),
    "大立光": ("3008.TW", "大立光"),
    "南亞科": ("2408.TW", "南亞科"),
    "旺宏": ("2337.TW", "旺宏"),
    "力積電": ("6770.TW", "力積電"),
    "華碩": ("2357.TW", "華碩"),
    "宏碁": ("2353.TW", "宏碁"),
    "世界": ("5347.TWO", "世界"),
    "兆豐金": ("2886.TW", "兆豐金"),
    "第一金": ("2892.TW", "第一金"),
}


def _load_external_tw_stock_map():
    """
    Optional extension point: drop `data/tw_stock_map.json` to expand the built-in mapping.
    Supported formats:
    1) {"公司名": ["2330.TW", "台積電"], ...}
    2) {"公司名": {"ticker": "2330.TW", "company_name": "台積電"}, ...}
    """
    if not EXTERNAL_STOCK_MAP_PATH.exists():
        return {}
    try:
        with EXTERNAL_STOCK_MAP_PATH.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    merged = {}

    def english_variants(name: str) -> list[str]:
        lowered = name.strip()
        if not lowered or not re.search(r"[A-Za-z]", lowered):
            return []
        variants = set()
        base = lowered
        variants.add(base)
        # common corporate suffixes
        cleaned = re.sub(r",?\s*(co\\.?|corp\\.?|corporation|inc\\.?|ltd\\.?|limited)\\s*$", "", base, flags=re.IGNORECASE)
        cleaned = re.sub(r"\\s+co\\.,?\\s*ltd\\.?\\s*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        if cleaned and cleaned != base:
            variants.add(cleaned)
        # strip punctuation for loose matching
        loose = re.sub(r"[^A-Za-z0-9\\s]", " ", cleaned).strip()
        loose = re.sub(r"\\s+", " ", loose)
        if loose and loose != cleaned:
            variants.add(loose)
        # Add short prefixes to support partial English queries (e.g. "Taiwan Semiconductor").
        words = loose.split()
        for n in (2, 3, 4):
            if len(words) > n:
                variants.add(" ".join(words[:n]))
        return sorted(variants, key=len, reverse=True)
    if isinstance(payload, dict):
        for name, value in payload.items():
            if not isinstance(name, str) or not name.strip():
                continue
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                ticker, canonical_name = value[0], value[1]
                if ticker and canonical_name:
                    key = name.strip()
                    merged[key] = (str(ticker), str(canonical_name))
                    for v in english_variants(key):
                        merged.setdefault(v, (str(ticker), str(canonical_name)))
            elif isinstance(value, dict):
                ticker = value.get("ticker")
                canonical_name = value.get("company_name") or name
                if ticker:
                    key = name.strip()
                    merged[key] = (str(ticker), str(canonical_name))
                    for v in english_variants(key):
                        merged.setdefault(v, (str(ticker), str(canonical_name)))
    return merged


_external_stock_map = _load_external_tw_stock_map()
COMMON_TW_STOCKS = {**_external_stock_map, **COMMON_TW_STOCKS}

TW_TICKER_TO_NAME: dict[str, str] = {}
for _, (ticker, canonical_name) in COMMON_TW_STOCKS.items():
    if ticker and canonical_name:
        TW_TICKER_TO_NAME[str(ticker).upper()] = str(canonical_name)
        TW_TICKER_TO_NAME[str(ticker).upper().replace(".TWO", "").replace(".TW", "")] = str(canonical_name)


def resolve_tw_company_name(ticker: str, fallback: str = "") -> str:
    """
    Resolve a Taiwan stock's Chinese short name whenever possible.
    Prefers: external stock map (data/tw_stock_map.json) -> built-in common list -> fallback.
    """
    t = (ticker or "").strip().upper()
    if not t:
        return fallback or ""
    code = t.replace(".TWO", "").replace(".TW", "")
    # Fast path: code key (e.g. "2312")
    if code in COMMON_TW_STOCKS:
        return COMMON_TW_STOCKS[code][1]
    # Value reverse map (e.g. "2312.TW" or "2312")
    if t in TW_TICKER_TO_NAME:
        return TW_TICKER_TO_NAME[t]
    if code in TW_TICKER_TO_NAME:
        return TW_TICKER_TO_NAME[code]
    return fallback or code


def sentiment_label_from_score(score: float) -> str:
    if score > 0.12:
        return "正面"
    if score < -0.08:
        return "負面"
    return "中立"

INVESTMENT_KEYWORDS = [
    "股票", "台股", "大盤", "加權", "投資", "買", "賣", "持有", "持股", "進場",
    "出場", "停損", "停利", "報酬", "走勢", "股價", "技術", "新聞", "情緒",
    "外資", "投信", "自營商", "三大法人", "融資", "融券",
]

FINANCIAL_TERMS = [
    # Common valuation / fundamentals
    "EPS", "每股盈餘",
    "本益比", "PER", "PE", "P/E",
    "本淨比", "PBR", "PB", "P/B", "股價淨值比",
    "殖利率", "股利", "股息", "現金股利", "股票股利",
    "ROE", "ROA", "營益率", "毛利率", "淨利率", "自由現金流", "FCF",
    "營收", "月營收", "年增", "季增", "YoY", "QoQ",
    "PEG", "Beta", "β", "Sharpe", "夏普", "波動", "標準差",
    # Technicals
    "K線", "K 線", "成交量", "量價", "均價", "均線", "月線", "季線", "年線",
    "RSI", "KD", "MACD",
    "乖離率", "布林通道", "Bollinger",
    "黃金交叉", "死亡交叉",
    "支撐", "壓力", "突破", "跌破", "回測", "反彈", "整理", "趨勢",
    # Trading / margin
    "停損", "停利", "風險報酬", "回撤", "Max DD", "最大回撤",
    "融資", "融券", "融資使用率", "券資比", "周轉率",
    # Our UI terms
    "量化訊號", "情緒分數", "建議可信度",
]

UNRELATED_KEYWORDS = [
    "天氣", "氣象", "溫度", "下雨",
    "寫程式", "程式", "coding", "bug",
    "食譜", "料理", "減肥",
    "旅遊", "機票", "住宿",
    "翻譯", "英文", "日文",
    "笑話", "梗圖",
    "電影", "音樂", "遊戲",
    "星座", "算命",
]

def find_stock_mentions(text):
    text = (text or "").strip()
    if not text:
        return []

    mentions = []
    seen = set()
    occupied_ranges: list[tuple[int, int]] = []

    def overlaps(existing_ranges, start, end):
        return any(not (end <= s or start >= e) for s, e in existing_ranges)

    # 1) Company-name mentions, ordered by first appearance and preferring longer matches.
    name_hits = []
    for company_name, (ticker, canonical_name) in COMMON_TW_STOCKS.items():
        pattern = re.escape(company_name)
        flags = re.IGNORECASE if re.search(r"[A-Za-z]", company_name) else 0
        for match in re.finditer(pattern, text, flags=flags):
            name_hits.append((match.start(), match.end(), len(company_name), ticker, canonical_name))
    for start, end, _, ticker, canonical_name in sorted(name_hits, key=lambda x: (x[0], -x[2])):
        if overlaps(occupied_ranges, start, end):
            continue
        if ticker in seen:
            continue
        mentions.append({"ticker": ticker, "company_name": canonical_name})
        seen.add(ticker)
        occupied_ranges.append((start, end))

    # 2) Numeric ticker mentions, ordered by appearance; prefer mapped exchange suffix.
    for match in re.finditer(r"(?<!\d)(\d{4,6})(?:\.(TW|TWO))?(?!\d)", text, re.IGNORECASE):
        if overlaps(occupied_ranges, match.start(), match.end()):
            continue
        code = match.group(1)
        suffix = match.group(2)
        if code in COMMON_TW_STOCKS:
            ticker, company_name = COMMON_TW_STOCKS[code]
        else:
            ticker = f"{code}.{suffix.upper()}" if suffix else f"{code}.TW"
            company_name = code
        if ticker in seen:
            continue
        mentions.append({"ticker": ticker, "company_name": company_name})
        seen.add(ticker)
        occupied_ranges.append((match.start(), match.end()))

    return mentions

def _load_api_cache():
    if not API_CACHE_PATH.exists():
        return {}
    try:
        with API_CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_api_cache(cache):
    try:
        API_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with API_CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ API 快取儲存失敗: {e}")

CACHE_SCHEMA_VERSION = "local-v4-complete-text"


def _api_cache_key(namespace, prompt):
    model_name = getattr(model_llm, "model", "llm")
    digest = hashlib.sha256(f"{CACHE_SCHEMA_VERSION}:{model_name}:{prompt}".encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"

def _get_cached_text(namespace, prompt, ttl_seconds):
    cache = _load_api_cache()
    key = _api_cache_key(namespace, prompt)
    item = cache.get(key)
    if not item:
        return None
    if time.time() - item.get("created_at", 0) > ttl_seconds:
        return None
    return item.get("text")

def _set_cached_text(namespace, prompt, text):
    cache = _load_api_cache()
    cache[_api_cache_key(namespace, prompt)] = {
        "created_at": time.time(),
        "text": text,
    }
    _save_api_cache(cache)

def cached_generate_json(namespace, prompt, ttl_seconds=86400):
    cached_text = _get_cached_text(namespace, prompt, ttl_seconds)
    if cached_text:
        return json.loads(cached_text)

    response = model_llm.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    raw = (response.text or "").strip()

    # Some models wrap JSON in markdown code fences — strip them
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

    if not raw:
        raise ValueError("LLM returned an empty response; cannot parse JSON")

    parsed = json.loads(raw)
    _set_cached_text(namespace, prompt, raw)
    return parsed

def cached_generate_text_stream(namespace, prompt, ttl_seconds=43200):
    cached_text = _get_cached_text(namespace, prompt, ttl_seconds)
    if cached_text:
        yield cached_text
        return

    chunks = []
    started_at = time.time()
    timed_out = False
    response = model_llm.generate_content(prompt, stream=True)
    for chunk in response:
        if time.time() - started_at > 120:
            if not chunks:
                raise TimeoutError("LLM 回覆生成逾時")
            timed_out = True
            timeout_note = "\n\n（分析生成時間較長，以下先提供目前可用重點；若需要更完整細節，可針對其中一段再追問。）"
            chunks.append(timeout_note)
            yield timeout_note
            break
        try:
            if chunk.text:
                chunks.append(chunk.text)
                yield chunk.text
        except ValueError:
            pass

    final_text = "".join(chunks)
    # Do not cache partial timeout output; otherwise users keep seeing the same truncated answer.
    if final_text and not timed_out:
        _set_cached_text(namespace, prompt, final_text)

def split_user_news_items(user_input):
    url_pattern = r"https?://[^\s，,。]+"
    urls = re.findall(url_pattern, user_input)
    text_without_urls = re.sub(url_pattern, "\n", user_input).strip()
    text_items = [
        item.strip(" \n\t-•")
        for item in re.split(r"\n{2,}|\n(?=\d+[.、])|\n(?=[-•])", text_without_urls)
        if item.strip(" \n\t-•")
    ]
    return urls, text_items

def is_user_provided_news_input(user_input):
    urls, text_items = split_user_news_items(user_input)
    if len(urls) >= 1:
        return True
    news_words = ["新聞", "快訊", "報導", "公告", "營收", "法說", "財報", "利多", "利空"]
    long_enough = len(user_input.strip()) >= 80 or "\n" in user_input
    return long_enough and any(word in user_input for word in news_words)

def fetch_article_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html = response.text
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else url
        body = re.sub(r"<(script|style).*?</\1>", " ", html, flags=re.I | re.S)
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
        return f"{title}\n{body[:1200]}"
    except Exception as e:
        print(f"⚠️ 使用者新聞網址讀取失敗: {url}, {e}")
        return url

def analyze_user_provided_news(user_input, target_company=None):
    urls, text_items = split_user_news_items(user_input)
    raw_items = [fetch_article_text_from_url(url) for url in urls] + text_items
    if not raw_items:
        raw_items = [user_input]

    details = []
    scores = []
    for idx, item in enumerate(raw_items[:8], start=1):
        clean_text = re.sub(r"\s+", " ", item).strip()
        score = _safe_sentiment_score(clean_text, target_company=target_company)
        pos_hits, neg_hits = _safe_extract_keywords(clean_text)
        if score >= 0.2:
            label = "偏多（利多）"
        elif score <= -0.2:
            label = "偏空（利空）"
        else:
            label = "中立"
        scores.append(score)
        details.append({
            "序號": idx,
            "摘要文字": clean_text[:160],
            "情緒判斷": label,
            "FinBERT分數": round(float(score), 4),
            "正面命中詞": pos_hits,
            "負面命中詞": neg_hits,
        })

    avg_score = round(float(np.mean(scores)), 4) if scores else 0.0
    if avg_score >= 0.2:
        overall_label = "偏多（利多）"
    elif avg_score <= -0.2:
        overall_label = "偏空（利空）"
    else:
        overall_label = "中立"

    return {
        "overall_label": overall_label,
        "sentiment_score": avg_score,
        "items": details,
    }

def generate_user_news_sentiment_answer_stream(user_input, analysis, investor_profile=None):
    investor_profile = investor_profile or {}
    details_text = json.dumps(analysis.get("items", []), ensure_ascii=False, indent=2)
    prompt = f"""
    你是一位貼近人性且專業的台股投資教練。使用者貼上了新聞網址或新聞內文，請你根據 FinBERT 情緒分數與新聞內容，判斷這則新聞對市場的「實際影響」，並給出客觀分析。
    請全程使用繁體中文，且回覆必須從使用者的角度出發，能夠真正解決他們的痛點，不要像冰冷的 AI 機器人或過度套用樣板。

    【使用者貼上的內容】：
    {user_input[:3000]}

    【情緒模型初步判定】：
    - 綜合判斷：{analysis.get('overall_label')}
    - 綜合 FinBERT 分數：{analysis.get('sentiment_score')}（範圍 -1 到 1，越正越偏利多，越負越偏利空）
    - 逐則細節：{details_text}

    【投資人設定】：
    - 投資風格：{investor_profile.get('style', '穩健')}
    - 最大可接受虧損：{investor_profile.get('max_loss_pct', 10)}%

    【分析與回覆要求】：
    1. **破題與情緒定調**：第一段用一句話白話總結這則新聞的整體情緒（偏多、偏空或中立），並解釋這個分數（{analysis.get('sentiment_score')}）代表什麼意思。
    2. **深度影響評估（有什麼影響）**：這是最重要的部分！請具體說明這則新聞「為什麼」會產生這種情緒分數，以及它對相關公司營收、未來發展或股價趨勢可能會造成「什麼實質影響」。
    3. **行動與應對建議**：以朋友或教練的口吻，結合投資人的「{investor_profile.get('style', '穩健')}」風格，給予如何看待這則新聞的建議（例如：這是短線雜音還是長線利多？需不需要立刻動作？）。
    
    ⚠️ *風險提示：金融市場有不確定性，本分析僅供參考。*
    """
    yield from cached_generate_text_stream("user_news_sentiment", prompt, ttl_seconds=21600)

def _static_financial_term_answer(user_input, investor_profile=None):
    investor_profile = investor_profile or {}
    text = (user_input or "").lower()
    style = investor_profile.get("style", "穩健")

    term_guides = {
        "本益比": (
            "本益比是什麼",
            "本益比可以想成「你願意付幾年的獲利，買下這家公司」。例如本益比 20 倍，粗略理解就是市場願意用約 20 年獲利的價格來買它。",
            "它常用來判斷股價相對獲利是貴還是便宜。高本益比不一定代表不能買，可能代表市場期待成長；低本益比也不一定便宜，可能是獲利正在下滑。",
            "看本益比時要和同產業、公司成長率、景氣循環一起看。對穩健投資人來說，本益比偏高時不要一次重押，最好等回檔或等獲利成長跟上估值。"
        ),
        "macd": (
            "MACD 是什麼",
            "MACD 是用兩條移動平均線的差距，判斷股價動能是變強還是變弱的技術指標。",
            "白話說，它像是在看車子的加速度：價格還在漲，不代表動能一定變強；MACD 可以幫你看漲勢是不是開始沒力。",
            "常見用法是看 DIF、MACD 柱狀體與零軸。由負翻正通常代表動能改善，但不能只靠 MACD 買賣，還要搭配均線、成交量與支撐壓力。"
        ),
        "rsi": (
            "RSI 是什麼",
            "RSI 是衡量一段時間內買盤或賣盤力道強弱的指標，可以想成股價短線熱度計。",
            "一般常用 30 和 70 當參考：RSI 高於 70 可能偏熱，低於 30 可能偏冷，但強勢股可能長時間維持高檔。",
            "穩健用法是不要看到 RSI 低就立刻買，而是搭配支撐、趨勢和量能確認是否真的止跌。"
        ),
        "kd": (
            "KD 是什麼",
            "KD 是觀察短線價格位置與轉折的技術指標，常用來看股價是否偏熱、偏冷，或是否出現轉強/轉弱訊號。",
            "K 線上穿 D 線常被稱為黃金交叉，可能代表短線轉強；K 線下穿 D 線則可能代表短線轉弱。",
            "KD 在盤整區比較有用，但在強趨勢行情容易出現假訊號，所以要搭配趨勢和成交量。"
        ),
        "roe": (
            "ROE 是什麼",
            "ROE 是股東權益報酬率，代表公司用股東投入的資本創造獲利的能力。",
            "可以把它想成公司使用資金的效率。ROE 越高通常代表經營效率越好，但也要確認不是靠高負債撐出來的。",
            "看 ROE 要搭配毛利率、負債比與現金流。穩健投資人適合找 ROE 穩定、不是只靠一次性收益拉高的公司。"
        ),
        "eps": (
            "EPS 是什麼",
            "EPS 是每股盈餘，代表公司賺到的錢平均分到每一股是多少。",
            "EPS 越高通常代表公司獲利能力越強，但要看它是持續成長，還是只有單季一次性暴衝。",
            "EPS 常和本益比一起看：股價除以 EPS 就是本益比。買股票時，不只看 EPS 高低，也要看未來能不能維持。"
        ),
        "殖利率": (
            "殖利率是什麼",
            "殖利率是股利相對股價的比例，可以想成你用現在價格買進後，股利帶來的現金回報率。",
            "例如股價 100 元、現金股利 5 元，殖利率就是 5%。",
            "高殖利率不一定安全，可能是股價下跌造成殖利率變高。穩健投資人要看公司配息是否穩定、現金流是否支撐得住。"
        ),
        "融資": (
            "融資使用率是什麼",
            "融資代表投資人借錢買股票，融資使用率越高，通常表示市場槓桿越重。",
            "如果股價下跌，融資部位容易被迫賣出，造成跌勢加劇。",
            "穩健投資人看到融資偏高時，要提高警覺，避免在籌碼過熱時追高。"
        ),
    }

    selected = None
    for key, guide in term_guides.items():
        aliases = [key]
        if key == "本益比":
            aliases += ["pe", "p/e", "per"]
        if key == "macd":
            aliases += ["macd"]
        if key == "rsi":
            aliases += ["rsi"]
        if key == "kd":
            aliases += ["kd"]
        if key == "roe":
            aliases += ["roe"]
        if key == "eps":
            aliases += ["eps", "每股盈餘"]
        if key == "殖利率":
            aliases += ["殖利率", "股息殖利率"]
        if key == "融資":
            aliases += ["融資", "融資使用率"]
        if any(alias.lower() in text for alias in aliases):
            selected = guide
            break

    if not selected:
        return None

    title, plain, application, caution = selected
    return (
        f"### {title}\n\n"
        f"**白話說明**：{plain}\n\n"
        f"**實際怎麼用**：{application}\n\n"
        f"**常見誤區**：不要只看單一數字就下買賣決策，最好同時搭配產業位置、趨勢、成交量與公司基本面。\n\n"
        f"**以「{style}」風格來看**：{caution}"
    )


def generate_financial_term_answer_stream(user_input, investor_profile=None):
    investor_profile = investor_profile or {}
    static_answer = _static_financial_term_answer(user_input, investor_profile)
    if static_answer:
        yield static_answer
        return

    prompt = f'''你是一位台股投資教練，擅長把複雜金融概念說得讓人聽懂、記得住。
請全程使用繁體中文，語氣自然，像在跟朋友說話。

【使用者問題】：{user_input}

【投資人設定】：
- 投資風格：{investor_profile.get('style', '穩健')}
- 最大可接受虧損：{investor_profile.get('max_loss_pct', 10)}%

━━━ 回覆準則 ━━━

**第一步：理解使用者真正想問什麼**
- 問單一名詞（如「RSI 是什麼？」）時，完整解釋這個詞。
- 問多個名詞的關係（如「RSI 和 MACD 有什麼差別？」）時，比較兩者並說明何時用哪個。
- 問如何在實際投資中應用（如「RSI 高於 70 要怎麼做？」）時，給出具體操作指引。
- 問題包含使用者的具體投資背景時，先回應背景，再解釋概念。
- 禁止說明你如何判斷問題，也禁止輸出任何內部分類字眼。

**第二步：輸出回覆**

必須包含：
1. **白話說明**：一句話說清楚這是什麼（不用學術定義）
2. **生活比喻**：用日常生活的比喻說明（如「就像體溫計，超過 37.5 就要注意」）
3. **實際應用**：這個指標/概念在投資時怎麼用，典型數值區間是什麼
4. **常見誤區**（如果有）：投資人最容易誤解的地方
5. **結合投資人設定**：考慮投資風格「{investor_profile.get('style', '穩健')}」，提醒何時特別需要注意此指標

⚠️ 技術指標注意：MACD、KD、RSI 等都是輔助工具，回覆中需提醒「不要單一指標做買賣決策」。
⚠️ 若問題很簡單（如「股利是什麼？」），不必強行寫 5 點，簡潔有力更好。
⚠️ 不要附加風險免責聲明，使用者知道這是教育內容。
⚠️ 禁止輸出任何 HTML 標籤（例如 <br>、<table>、<div>）。只用 Markdown（換行請用空行或 Markdown 語法）。
'''
    yield from cached_generate_text_stream("financial_term", prompt, ttl_seconds=86400)


# ==========================================
# 使用 LLM 進行「情緒污染過濾」
# ==========================================
def filter_pure_news_with_rules(company_name, news_list, aliases=None):
    if not news_list or company_name in ["台股", "大盤", "未知"]:
        return news_list
    entity_terms = [company_name, *(aliases or [])]
    entity_terms = [
        str(term).strip()
        for term in entity_terms
        if str(term).strip() and not re.fullmatch(r"\d{4,6}", str(term).strip())
    ]

    market_noise = ["台股", "大盤", "盤中", "盤後", "加權", "上市櫃", "三大法人"]
    low_value_noise = [
        "公司資料", "基本資料", "個股速覽", "權證", "購01", "購02", "購03", "售01", "售02", "售03",
        "財務報表", "營收表", "資產負債表", "現金流量表"
    ]
    unrelated_noise = ["商圈", "套房", "景觀宅", "租屋", "降價了", ".HK", "港股", "房市", "建案"]
    finance_keywords = [
        "股", "財報", "營收", "法說", "董事會", "EPS", "本益比", "毛利", "淨利", "資本支出",
        "投資", "法人", "外資", "漲", "跌", "買", "賣", "配息", "殖利率", "訂單", "產能", "公告"
    ]
    filtered = []
    for title in news_list:
        core_title = title.rsplit(' - ', 1)[0].strip()
        # Strict: must explicitly mention the target company in the title.
        if not any(term in core_title for term in entity_terms):
            continue
        # Drop obvious market-wide noise even if the company name appears.
        if any(word in core_title for word in market_noise):
            continue
        if any(word in core_title for word in low_value_noise):
            continue
        if any(word in core_title for word in unrelated_noise):
            continue
        if not any(word in core_title for word in finance_keywords):
            continue

        filtered.append(title)
    return filtered

def filter_pure_news_with_ai(company_name, news_list, aliases=None):
    if not news_list or company_name in ["台股", "大盤", "未知"]:
        return news_list
    if not USE_AI_NEWS_FILTER:
        return filter_pure_news_with_rules(company_name, news_list, aliases)
        
    try:
        refined_news = []
        for batch_start in range(0, min(len(news_list), MAX_SENTIMENT_NEWS), 20):
            batch = news_list[batch_start:batch_start + 20]
            news_text = "\n".join([f"[{i}] {title}" for i, title in enumerate(batch)])
            alias_text = "、".join(aliases or []) or "無"
            prompt = f"""
            你是一個嚴格的金融新聞守門員與實體情緒拆解專家。使用者目前只想分析【{company_name}】的「獨立情緒」。
            可接受的可信別名：{alias_text}
            請對以下新聞清單進行「結構化推論 (Chain of Thought)」審查，並判斷是否保留該新聞。

            【審查規則】：
            1. 實體關聯性（嚴格）：如果【{company_name}】或可信別名沒有出現在標題中，請一律丟棄 (drop)。標題中若只是剛好出現股票代號數字，也不能視為命中。
            2. 多實體雜訊：如同時出現多檔個股或大盤，原則上丟棄。但若主題明顯聚焦於【{company_name}】，請保留，並在 reason 補充說明。
            3. 低價值資訊：若標題是權證、公司資料、基本資料、個股速覽，或只是商品/衍生性金融商品名稱，即使提到【{company_name}】也要丟棄。
            4. 絕對靜止原則：⚠️ 絕對禁止改寫新聞標題！如判斷為保留 (keep)，必須 100% 輸出原始標題。

            請以 JSON Array 格式輸出，陣列中的每個物件需包含以下結構：
            [
              {{
                "original_title": "在此放入原始標題",
                "action": "keep 或 drop",
                "reason": "簡短說明保留或丟棄的判斷理由"
              }}
            ]

            新聞標題列表：
            {news_text}
            """
            reasoning_results = cached_generate_json("news_filter", prompt, ttl_seconds=86400)
            refined_news.extend(item["original_title"] for item in reasoning_results if item.get("action") == "keep")
        return refined_news
    except Exception as e:
        print(f"⚠️ 過濾與提煉失敗，啟動嚴格字串比對 fallback: {e}")
        return filter_pure_news_with_rules(company_name, news_list, aliases)

@ttl_cache_data(ttl=1800)
def fetch_stock_or_macro_sentiment(ticker, company_name, days=5):
    print("📰 使用本地端新聞抓取：Yahoo 股市 RSS + Google News RSS + GDELT 備援。")

    resolved_name = resolve_tw_company_name(str(ticker or ""), str(company_name or ""))
    if resolved_name and not re.fullmatch(r"\d{4,6}", resolved_name):
        company_name = resolved_name

    if company_name and re.fullmatch(r"\d{4,6}", str(company_name).strip()):
        return {
            "news_summary": f"近 {days} 天內無相關專屬新聞。",
            "sentiment_score": 0.0,
            "raw_news": [],
            "news_count_status": "none"
        }

    macro_entities = {"台股", "大盤", "加權指數", "加權"}
    is_macro = (not company_name) or (company_name == "未知") or (company_name in macro_entities)

    # For macro queries, we intentionally broaden recall and do not use exact quoted match.
    # For single-stock queries, we keep the strict entity rule (quoted company name).
    if is_macro:
        search_keyword = "台股 OR 加權指數 OR 加權"
    else:
        search_keyword = company_name
    
    def collect_aliases():
        if is_macro:
            return []
        code = str(ticker or "").upper().replace(".TWO", "").replace(".TW", "")
        aliases = []
        for name, (mapped_ticker, canonical_name) in COMMON_TW_STOCKS.items():
            mapped_code = str(mapped_ticker).upper().replace(".TWO", "").replace(".TW", "")
            if mapped_code != code:
                continue
            candidate = str(name).strip()
            if not candidate or candidate == company_name or re.fullmatch(r"\d{4,6}", candidate):
                continue
            if len(candidate) > 40:
                continue
            aliases.append(candidate)
            if len(aliases) >= 4:
                break
        return aliases

    trusted_aliases = collect_aliases()

    def news_search_query(query_str, exact=True):
        # 嚴格排除任何討論區、論壇、農場文，確保新聞來源為正規財經新聞
        # 個股：給公司名稱加上雙引號，強制精準搜尋
        # 大盤：採用 OR 擴充（避免 "大盤" 太難搜到真正台股市場新聞）
        if is_macro:
            return f'({query_str}) -site:cmoney.tw -同學會 -討論 -PTT -Dcard -Mobile01 -社團 -貼文 -懶人包'
        if exact:
            return f'"{query_str}" -site:cmoney.tw -同學會 -討論 -PTT -Dcard -Mobile01 -社團 -貼文 -懶人包'
        return f'{query_str} -site:cmoney.tw -同學會 -討論 -PTT -Dcard -Mobile01 -社團 -貼文 -懶人包'
    
    def get_google_news(query_str, exact=True):
        advanced_query = news_search_query(query_str, exact=exact)
        query = urllib.parse.quote(advanced_query)
        url = f"https://news.google.com/rss/search?q={query}+when:{days}d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"

        def fetcher():
            text = _fetch_news_url(url, f"Google News RSS ({query_str})", timeout=4, retries=2)
            if text is None:
                return None
            try:
                root = ET.fromstring(text)
                return [item.find('title').text for item in root.findall('.//item') if item.find('title') is not None]
            except Exception as exc:
                print(f"❌ Google News RSS 解析失敗 ({query_str})！: {exc}")
                return None

        return _cached_news_source("google_news", [query_str, exact, days, is_macro], fetcher)

    def get_gdelt_news(query_str, exact=True):
        if is_macro:
            gdelt_query = '(台股 OR 加權指數 OR 加權)'
        elif exact:
            gdelt_query = f'"{query_str}"'
        else:
            gdelt_query = query_str
        query = urllib.parse.quote(gdelt_query)
        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc"
            f"?query={query}&mode=artlist&format=json&sort=datedesc"
            f"&maxrecords={MAX_SENTIMENT_NEWS}&timespan={max(1, int(days))}d"
        )

        def fetcher():
            text = _fetch_news_url(url, f"GDELT DOC API ({query_str})", timeout=8, retries=2)
            if text is None:
                return None
            try:
                payload = json.loads(text)
            except Exception as exc:
                print(f"❌ GDELT DOC API 解析失敗 ({query_str})！: {exc}")
                return None
            articles = payload.get("articles") if isinstance(payload, dict) else None
            if not isinstance(articles, list):
                return []
            titles = []
            for article in articles:
                if not isinstance(article, dict):
                    continue
                title = str(article.get("title") or "").strip()
                if not title:
                    continue
                domain = str(article.get("domain") or "").strip()
                titles.append(f"{title} - {domain}" if domain else title)
            return titles

        return _cached_news_source("gdelt", [query_str, exact, days, is_macro], fetcher)

    def get_yahoo_news():
        code = str(ticker or "").upper().replace(".TWO", "").replace(".TW", "")
        url = "https://tw.stock.yahoo.com/rss?category=tw-market" if is_macro else f"https://tw.stock.yahoo.com/rss?s={code}"

        def fetcher():
            text = _fetch_news_url(url, f"Yahoo 股市 RSS ({code or 'market'})", timeout=6, retries=2)
            if text is None:
                return None
            try:
                root = ET.fromstring(text)
            except Exception as exc:
                print(f"❌ Yahoo 股市 RSS 解析失敗 ({code or 'market'})！: {exc}")
                return None

            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            titles = []
            for item in root.findall(".//item"):
                title = item.findtext("title")
                if not title:
                    continue
                pub_date = item.findtext("pubDate")
                if pub_date:
                    try:
                        parsed = parsedate_to_datetime(pub_date)
                        if parsed.tzinfo is None:
                            parsed = parsed.replace(tzinfo=timezone.utc)
                        if parsed < cutoff:
                            continue
                    except Exception:
                        continue
                titles.append(title)
            return titles

        return _cached_news_source("yahoo_rss", [code, days, is_macro], fetcher)

    def clean_and_dedup(titles):
        import difflib
        deduped_titles = []
        for raw_title in titles:
            # 1. 移除 Google 新聞後方的媒體名稱 (如 " - Yahoo奇摩股市")
            core_title = raw_title.rsplit(' - ', 1)[0].strip()
            # 2. 移除常見的前後綴雜訊，以利比對真實字義 (包含《》、【】、〈〉等各種括號)
            clean_text = re.sub(r'^.*?》|《.*?》|【.*?】|〈.*?〉|「.*?」|『.*?』|快訊／|- 上市櫃', '', core_title).strip()
            
            # 3. 捨棄字數過少的新聞（例如：僅有股票名字或「盤中速報」等無效資訊）
            if len(clean_text) < 7:
                continue
            
            is_dup = False
            for seen in deduped_titles:
                seen_clean = re.sub(r'^.*?》|《.*?》|【.*?】|〈.*?〉|「.*?」|『.*?』|快訊／|- 上市櫃', '', seen.rsplit(' - ', 1)[0]).strip()
                
                # 規則 A：使用 difflib 計算字串相似度 (大於35%視為重複)
                similarity = difflib.SequenceMatcher(None, clean_text, seen_clean).ratio()
                
                # 規則 B：開頭前 5 個字完全相同，視為同系列事件報導
                prefix_len = 5
                same_prefix = False
                if len(clean_text) >= prefix_len and len(seen_clean) >= prefix_len:
                    if clean_text[:prefix_len] == seen_clean[:prefix_len]:
                        same_prefix = True
                
                # 只要符合任一規則即視為重複新聞，不予採用
                if similarity > 0.35 or same_prefix:
                    is_dup = True
                    break
            
            if not is_dup:
                deduped_titles.append(raw_title)
        return deduped_titles

    # 取消過多 print，保持終端機乾淨
    if is_macro:
        query_plan = [(search_keyword, False)]
    else:
        query_plan = [
            (company_name, True),
            (f"{company_name} 股票", False),
            *[(alias, True) for alias in trusted_aliases],
        ]

    stock_titles = []
    had_fetch_success = False

    yahoo_titles = get_yahoo_news()
    if yahoo_titles is not None:
        had_fetch_success = True
        stock_titles.extend(yahoo_titles)
        print(f"📰 Yahoo RSS 成功：{len(yahoo_titles)} 則")
    else:
        print("📰 Yahoo RSS 失敗")

    for query_text, exact in query_plan:
        if len(stock_titles) >= MAX_SENTIMENT_NEWS:
            break
        titles = get_google_news(query_text, exact=exact)
        if titles is None:
            continue
        had_fetch_success = True
        stock_titles.extend(titles)
        print(f"📰 Google News RSS 成功：+{len(titles)} 則（query={query_text}）")
        if len(stock_titles) >= MAX_SENTIMENT_NEWS * 2:
            break

    for query_text, exact in query_plan:
        if len(stock_titles) >= MAX_SENTIMENT_NEWS:
            break
        titles = get_gdelt_news(query_text, exact=exact)
        if titles is None:
            continue
        had_fetch_success = True
        stock_titles.extend(titles)
        print(f"📰 GDELT 備援成功：+{len(titles)} 則（query={query_text}）")
        if len(stock_titles) >= MAX_SENTIMENT_NEWS * 2:
            break

    if not had_fetch_success:
        print("📰 新聞抓取最終失敗：所有來源皆無法連線或解析")
        return {
            "news_summary": "新聞抓取失敗：本地新聞來源暫時無法取得，請稍後重試。",
            "sentiment_score": 0.0,
            "raw_news": [],
            "news_count_status": "fetch_failed",
        }
    unique_news = clean_and_dedup(stock_titles)
    if not is_macro:
        # Only run strict entity filtering for single-stock queries.
        unique_news = filter_pure_news_with_ai(company_name, unique_news, trusted_aliases)
    unique_news = unique_news[:MAX_SENTIMENT_NEWS]
    
    # 根據取得的新聞數量決定狀態與總結字首
    news_count = len(unique_news)
    
    if news_count == 0:
        return {
            "news_summary": f"近 {days} 天內無相關專屬新聞。", 
            "sentiment_score": 0.0, 
            "raw_news": [],
            "news_count_status": "none"
        }
    elif news_count < 5:
        summary_prefix = f"近期相關新聞較少 (僅 {news_count} 篇)，請以技術面與量化數據為主"
        news_count_status = "few"
    else:
        summary_prefix = "抓取【台股/加權指數】市場新聞" if is_macro else f"抓取【{search_keyword}】專屬純淨新聞"
        news_count_status = "sufficient"

    recent_news_titles = unique_news[:MAX_SENTIMENT_NEWS]
    
    scores = []
    news_details = [] 
    for title in recent_news_titles:
        core_title = title.rsplit(' - ', 1)[0].strip() 
        # 對最終顯示的標題也進行嚴格清洗，避免保留【台股盤後】、《ESG》這類干擾AI的字眼
        clean_title = re.sub(r'^.*?》|《.*?》|【.*?】|〈.*?〉|「.*?」|『.*?』|快訊／|- 上市櫃', '', core_title).strip()
        
        # 修正：傳入公司名稱，讓分數只對應該公司的段落
        # "南亞科漲停打開、旺宏亮綠燈" -> 只算 "旺宏亮綠燈"
        score = _safe_sentiment_score(clean_title, target_company=company_name)
        
        scores.append(score)
        news_details.append({
            "新聞標題": clean_title, 
            "FinBERT分數": score,
            "情緒標籤": sentiment_label_from_score(score),
        })
        
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    
    first_clean_title = news_details[0]["新聞標題"] if news_details else "無相關新聞"
    summary = f"近 {days} 天綜合抓取 {len(recent_news_titles)} 則 ({summary_prefix})。最新頭條：「{first_clean_title}」"
    
    return {
        "news_summary": summary, 
        "sentiment_score": avg_score,
        "raw_news": news_details,
        "news_count_status": news_count_status
    }

def fetch_realtime_stock_data(ticker):
    try:
        if not ticker or ticker == '未知' or ticker == '未知標的':
            return {"latest_price": "無", "trend_summary": "無資料"}, None

        raw = str(ticker).strip().upper()
        code = raw.replace(".TWO", "").replace(".TW", "")
        candidates = []
        if raw == "^TWII":
            candidates = ["^TWII"]
        elif code in COMMON_TW_STOCKS:
            candidates.append(COMMON_TW_STOCKS[code][0].upper())
        if raw and raw not in candidates:
            candidates.append(raw)
        if code and f"{code}.TW" not in candidates:
            candidates.append(f"{code}.TW")
        if code and f"{code}.TWO" not in candidates:
            candidates.append(f"{code}.TWO")

        df = pd.DataFrame()
        resolved_ticker = raw
        latest_price = None
        def normalize_history_frame(frame):
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                return pd.DataFrame()
            normalized = frame.copy()
            if isinstance(normalized.columns, pd.MultiIndex):
                normalized.columns = [
                    col[0] if isinstance(col, tuple) and len(col) > 0 else str(col)
                    for col in normalized.columns
                ]
            for required in ["Open", "High", "Low", "Close"]:
                if required not in normalized.columns:
                    return pd.DataFrame()
            return normalized

        for candidate in candidates:
            stock = yf.Ticker(candidate)
            for period in ["2y", "1y", "6mo", "3mo", "1mo"]:
                try:
                    df = normalize_history_frame(stock.history(period=period, auto_adjust=False))
                except Exception:
                    df = pd.DataFrame()
                if not df.empty:
                    resolved_ticker = candidate
                    break
                try:
                    download_df = normalize_history_frame(
                        yf.download(candidate, period=period, progress=False, auto_adjust=False, threads=False)
                    )
                except Exception:
                    download_df = pd.DataFrame()
                if not download_df.empty:
                    df = download_df
                    resolved_ticker = candidate
                    break
            if isinstance(df, pd.DataFrame) and not df.empty:
                break
            try:
                fast_info = getattr(stock, "fast_info", None) or {}
                latest_price = fast_info.get("lastPrice") or fast_info.get("regularMarketPrice")
            except Exception:
                latest_price = latest_price

        if df.empty and latest_price is None:
            return {"latest_price": "無", "trend_summary": "無資料"}, None

        df = normalize_history_frame(df).dropna(subset=['Close'])
        if df.empty:
            if latest_price is None:
                return {"latest_price": "無", "trend_summary": "無資料"}, None
            return {
                "latest_price": round(float(latest_price), 2),
                "trend_summary": "資料不足",
                "resolved_ticker": resolved_ticker.replace('.TWO', '').replace('.TW', ''),
            }, None

        latest_price = round(float(df['Close'].iloc[-1]), 2)
        # 取近 20 天均線做為趨勢指標
        ma20 = df['Close'].tail(20).mean()
        trend = "站上月線" if latest_price > ma20 else "跌破月線"

        return {"latest_price": latest_price, "trend_summary": trend, "resolved_ticker": resolved_ticker.replace('.TWO', '').replace('.TW', '')}, df
    except: return {"latest_price": "未知", "trend_summary": "抓取失敗"}, None

def run_quant_model(ticker, df_history, intent):
    if df_history is None or df_history.empty:
        return {
            "predicted_return": 0.0,
            "signal": "無法預測",
            "regime": "無資料",
            "max_dd": "N/A",
            "model_name": "stock_tech2_regime_embedding_h2_t10.pth",
            "model_count": 0,
            "error_message": "df_history 為空，無法執行量化模型。",
        }

    if not ENABLE_QUANT_MODEL:
        return _lightweight_quant_fallback(df_history)

    # 移除尚未開盤導致的 NaN 收盤價
    df = df_history.dropna(subset=['Close']).copy()
    if 'Date' not in df.columns:
        df['Date'] = pd.to_datetime(df.index)
    df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce').dt.tz_localize(None)
    df = df.reset_index(drop=True)
    if len(df) < 60:
        return {
            "predicted_return": 0.0,
            "signal": "資料過少",
            "regime": "未知",
            "max_dd": "N/A",
            "model_name": "stock_tech2_regime_embedding_h2_t10.pth",
            "model_count": 0,
            "error_message": f"歷史資料只有 {len(df)} 筆，少於模型需要的 60 筆。",
        }
        
    try:
        torch, get_config, FeatureEngineer, RegimeEmbeddingLSTM = _load_quant_dependencies()
        config = get_config()
        engineer = FeatureEngineer(config)
        
        # 1. 基礎報酬計算
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['log_return_5d'] = np.log(df['Close'] / df['Close'].shift(5))
        df['log_return_10d'] = np.log(df['Close'] / df['Close'].shift(10))
        df['log_return_20d'] = np.log(df['Close'] / df['Close'].shift(20))
        
        # 2. 自動加入各項技術指標與上下文
        df = engineer.add_all_features(df)
        
        # 3. 計算動態最大回撤 (Drawdown) 用以定義 Regime
        df['rolling_peak'] = df['Close'].expanding().max()
        df['drawdown'] = (df['Close'] - df['rolling_peak']) / df['rolling_peak']
        latest_dd = df['drawdown'].iloc[-1]
        
        # 計算近一年 (252 個交易日) 的「真實最大回撤 (Max Drawdown)」 供介面顯示
        past_year_close = df['Close'].tail(252)
        rolling_max_1y = past_year_close.expanding().max()
        drawdowns_1y = (past_year_close - rolling_max_1y) / rolling_max_1y
        max_dd_1y = drawdowns_1y.min()
        
        # 4. 使用 summary.csv 依 Sharpe/CAGR、Direction Acc、Max DD 挑出的最佳 regime embedding 模型。
        #    stock_tech2_regime_embedding_h2_t10:
        #    Sharpe=2.796, CAGR=59.16%, Direction Acc=63.46%, Max DD=-6.70%
        model_portfolio = [
            ("stock_tech2_regime_embedding_h2_t10.pth", 10, -0.10, "stock_tech2", 2),
        ]
        signal_threshold = 0.004
        
        model_dir = Path(os.path.join(_project_root, "experiments", "saved_models"))
        seq_len = getattr(config.data, 'lookback_window', 60)
        
        predictions = []
        model_details = []
        for w_name, n_feat, t_thresh, combo, h_hor in model_portfolio:
            w_path = model_dir / w_name
            if not w_path.exists():
                print(f"⚠️ 找不到權重: {w_name}")
                continue
                
            # 準備該模型的特徵切片並做滾動標準化
            f_set = MODEL_FEATURE_SETS.get(combo)
            if f_set is None:
                f_set = engineer.get_feature_set(combo)
            df_norm = normalize_features_rolling_for_inference(df, f_set, window=252)
            
            # 取最後 60 天序列
            feats = df_norm[f_set].values[-seq_len:]
            if len(feats) < seq_len or np.isnan(feats).any():
                 print(f"⚠️ 特徵序列不完整或存在 NaN: {combo}")
                 continue
                 
            x_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            
            # 定義目前市場狀態 Regime Label (0: 牛市, 1: 熊市)
            r_label = 1 if latest_dd <= t_thresh else 0
            r_t = torch.tensor([r_label], dtype=torch.long)
            
            # 初始化模型與載入權重
            model = RegimeEmbeddingLSTM(
                n_features=n_feat,
                hidden_size=getattr(config.model, 'd_model', 64),
                num_layers=getattr(config.model, 'n_encoder_layers', 3),
                dropout=getattr(config.model, 'dropout', 0.1),
                seq_len=seq_len,
                n_regimes=2,
                use_attention=True
            )
            
            state = torch.load(w_path, map_location=torch.device('cpu'))
            state_dict = state['model_state_dict'] if 'model_state_dict' in state else state
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            # 進行前瞻推論
            with torch.no_grad():
                out = model(x_t, r_t)
                
            # 模型原始輸出對應 horizon 期報酬訊號；對外一律轉為一般報酬率。
            pred_log = out[0].item() if isinstance(out, tuple) else out.item()
            period_pred = np.exp(pred_log) - 1
            daily_pred = np.exp(pred_log / h_hor) - 1
            predictions.append(daily_pred)
            model_details.append({
                "model": w_name,
                "feature_combo": combo,
                "horizon_days": h_hor,
                "regime_label": "熊市" if r_label == 1 else "牛市/整理",
                "period_predicted_return": round(float(period_pred), 6),
                "daily_predicted_return": round(float(daily_pred), 6),
            })
            
        if not predictions:
            return {
                "predicted_return": 0.0,
                "signal": "集成失敗",
                "regime": "缺少模型",
                "max_dd": f"({max_dd_1y*100:.1f}%)",
                "model_name": "stock_tech2_regime_embedding_h2_t10.pth",
                "model_count": 0,
                "error_message": "模型權重不存在、特徵序列不足，或特徵標準化後仍含 NaN。",
            }
            
        # 5. 輸出模型的預期數值
        avg_ret = float(np.mean(predictions))
        
        # 制定交易決策邏輯：採用該模型在 summary.csv 的 Best Threshold。
        if avg_ret > signal_threshold: signal = "買入"
        elif avg_ret < -signal_threshold: signal = "賣出"
        else: signal = "觀望"
             
        # 趨勢標籤
        if latest_dd <= -0.15: regime = "趨勢轉弱 (熊市)"
        elif latest_dd <= -0.05: regime = "回檔整理"
        else: regime = "強勢偏多"
             
        return {
            "predicted_return": avg_ret,
            "signal": signal,
            "regime": regime,
            "max_dd": f"({max_dd_1y*100:.1f}%)",
            "model_name": "stock_tech2_regime_embedding_h2_t10.pth",
            "model_details": model_details,
            "model_count": len(predictions),
        }
        
    except Exception as e:
        print(f"❌ 嚴重錯誤發生在 run_quant_model: {e}")
        import traceback
        traceback.print_exc()
        return {
            "predicted_return": 0.0,
            "signal": "運算錯誤",
            "regime": "未知",
            "max_dd": "N/A",
            "model_name": "stock_tech2_regime_embedding_h2_t10.pth",
            "model_count": 0,
            "error_message": f"{type(e).__name__}: {e}",
        }

def extract_intent_with_rules(user_input, default_horizon, default_risk):
    text = user_input.strip()
    if not text:
        return None

    has_financial_term = any(term.lower() in text.lower() for term in FINANCIAL_TERMS)
    has_investment_context = any(keyword in text for keyword in INVESTMENT_KEYWORDS) or has_financial_term

    if is_user_provided_news_input(text):
        return {
            "ticker": None,
            "company_name": "使用者提供新聞",
            "intent_type": "USER_NEWS",
            "horizon": default_horizon,
            "risk_tolerance": default_risk,
            "intent_source": "rule",
        }

    # Financial terms without an explicit stock: always answer as a term explanation.
    # (If the user wants a specific stock, they can add ticker/company name.)
    if has_financial_term and not re.search(r"(?<!\d)(\d{4})(?:\.(TW|TWO))?(?!\d)", text, re.IGNORECASE):
        mentioned_stock = any(company_name in text for company_name in COMMON_TW_STOCKS)
        if not mentioned_stock:
            return {
                "ticker": None,
                "company_name": "金融術語",
                "intent_type": "FINANCIAL_TERM",
                "horizon": default_horizon,
                "risk_tolerance": default_risk,
                "intent_source": "rule",
            }

    stock_mentions = find_stock_mentions(text)
    if stock_mentions:
        if len(stock_mentions) >= 2 and any(word in text for word in ["比較", "哪個", "誰", "vs", "VS", "差異", "選"]):
            return {
                "ticker": None,
                "company_name": "多檔比較",
                "intent_type": "STOCK_COMPARE",
                "stocks": stock_mentions,
                "horizon": default_horizon,
                "risk_tolerance": default_risk,
                "intent_source": "rule",
            }
        first_stock = stock_mentions[0]
        return {
            "ticker": first_stock["ticker"],
            "company_name": first_stock["company_name"],
            "intent_type": "STOCK",
            "horizon": default_horizon,
            "risk_tolerance": default_risk,
            "intent_source": "rule",
        }

    ticker_match = re.search(r'(?<!\d)(\d{4})(?:\.(TW|TWO))?(?!\d)', text, re.IGNORECASE)
    if ticker_match:
        market_suffix = ticker_match.group(2)
        ticker = f"{ticker_match.group(1)}.{market_suffix.upper()}" if market_suffix else f"{ticker_match.group(1)}.TW"
        return {
            "ticker": ticker,
            "company_name": resolve_tw_company_name(ticker, ticker_match.group(1)),
            "intent_type": "STOCK",
            "horizon": default_horizon,
            "risk_tolerance": default_risk,
            "intent_source": "rule",
        }

    if any(keyword in text for keyword in ["大盤", "台股", "加權指數", "加權"]) and not any(kw in text for kw in ["比較", "個股"]):
        return {
            "ticker": "^TWII",
            "company_name": "大盤",
            "intent_type": "MARKET",
            "horizon": default_horizon,
            "risk_tolerance": default_risk,
            "intent_source": "rule",
        }

    # Unrelated or small talk: avoid stock lookup attempts.
    # If it doesn't look like Taiwan stock/investment context, treat as unrelated.
    if (not has_investment_context and not any(keyword in text for keyword in ["大盤", "台股", "加權指數", "加權"])) or any(keyword in text for keyword in UNRELATED_KEYWORDS):
        return {
            "ticker": None,
            "company_name": "未知",
            "intent_type": "UNRELATED",
            "horizon": default_horizon,
            "risk_tolerance": default_risk,
            "intent_source": "rule",
        }

    return None

def extract_intent(user_input, default_horizon, default_risk, chat_history):
    rule_intent = extract_intent_with_rules(user_input, default_horizon, default_risk)
    if rule_intent:
        return rule_intent

    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-3:]])
    deterministic_mentions = find_stock_mentions(user_input)
    prompt = f"""
    你是一個專業的台灣金融意圖解析器。請根據【歷史對話】與【最新輸入】判斷使用者的需求。

    請嚴格遵守以下邏輯：
    1. 如果使用者問的是「大盤」、「台股」、「加權指數」，或是「詢問最近股價/投資相關問題但沒有講到具體標的」，請一律歸類為大盤，將 "ticker" 設為 "^TWII"，"company_name" 設為 "大盤"。
    2. 如果有明確指名個股，"ticker" 必須是 Yahoo Finance 格式（例如: 2330.TW, 8069.TWO），"company_name" 為其中文簡稱。
    3. 如果使用者明確想比較兩檔或以上股票，請將 "intent_type" 設為 "STOCK_COMPARE"，並輸出 stocks 陣列（每個元素包含 ticker 與 company_name）。
       - stocks 必須包含使用者輸入中出現的所有標的，不可漏掉任何一檔。
       - 若下方【偵測到的標的 (Deterministic)】不為空，stocks 必須完整包含這些標的；不得自行刪減或替換為其他代號。
    4. 如果使用者詢問 EPS、本益比、殖利率、MACD、KD、RSI、ROE、最大回撤等金融專有名詞，且沒有指定個股，請將 "intent_type" 設為 "FINANCIAL_TERM"，"ticker" 設為 null。
    5. 如果使用者貼上新聞網址或新聞內文，想判斷新聞情緒，請將 "intent_type" 設為 "USER_NEWS"，"ticker" 設為 null。
    6. ⚠️ 如果使用者輸入的問題「完全與台股、投資、理財、財經無關」（例如平常閒聊、問天氣、寫程式等），請強制將 "intent_type" 設為 "UNRELATED"，"ticker" 設為 null。

    請以 JSON 格式輸出：
    {{
      "ticker": "字串或 null",
      "company_name": "中文名稱 或 大盤",
      "intent_type": "MARKET", "STOCK", "STOCK_COMPARE", "FINANCIAL_TERM", "USER_NEWS" 或 "UNRELATED",
      "stocks": [{{"ticker": "字串", "company_name": "中文名稱"}}],
      "horizon": 1 或 20,
      "risk_tolerance": "low", "medium", 或 "high"
    }}

    【歷史對話】：{history_text}
    【最新輸入】：{user_input}
    【偵測到的標的 (Deterministic)】：{deterministic_mentions}
    """
    try:
        intent = cached_generate_json("intent", prompt, ttl_seconds=86400)
        intent["intent_source"] = "groq_cached"
        return intent
    except Exception as e:
        print(f"❌ Groq LLM API 發生錯誤: {e}")
        return {"ticker": "未知標的", "company_name": "未知", "horizon": default_horizon, "risk_tolerance": default_risk}

def generate_investment_advice_stream(user_input, intent, quant_data, stock_data, news_data, chat_history, investor_profile=None):
    raw_news = news_data.get('raw_news') or []
    pos_headlines = [n['新聞標題'] for n in raw_news if n.get('情緒標籤') == '正面'][:3]
    neg_headlines = [n['新聞標題'] for n in raw_news if n.get('情緒標籤') == '負面'][:3]
    investor_profile = investor_profile or intent.get("investor_profile", {})

    news_status_prompt = ""
    status = news_data.get("news_count_status", "sufficient")
    if status == "none":
        news_status_prompt = "⚠️ 近期無相關新聞，情緒分數不具代表性，請完全依賴量化模型數據，不要編造新聞理由。"
    elif status == "few":
        news_status_prompt = "⚠️ 相關新聞數量偏少（不足5篇），請在建議中說明情緒分析代表性不足，判斷權重向量化數據傾斜。"

    history_summary = ""
    if chat_history:
        recent = [m for m in chat_history[-6:] if m.get("role") in ("user", "assistant")]
        if recent:
            history_summary = "\n".join(f"{m['role'].upper()}: {m['content'][:200]}" for m in recent)

    prompt = f'''你是一位具備量化與基本面背景的台股投資顧問，同時也是一位能說人話的財務教練。
請全程使用繁體中文。你的核心任務是：**讓使用者真正理解現在該怎麼做，以及為什麼**。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【使用者問題】
{user_input}

【對話背景（最近幾輪）】
{history_summary if history_summary else "（無歷史對話）"}

【分析標的】
- 公司：{intent.get('company_name', '未知')} ({intent.get('ticker', '未知')})
- 目前股價：{stock_data.get('latest_price', '未知')}

【量化模型輸出】
- 訊號：{quant_data.get('signal', '未知')}（買入 / 觀望 / 賣出）
- 預期報酬：{round(quant_data.get('predicted_return', 0) * 100, 3)}%（基於歷史價量，非新聞驅動）
- 市場狀態 (Regime)：{quant_data.get('regime', '未知')}
- 近一年最大回撤：{quant_data.get('max_dd', 'N/A')}
- 最佳模型：{quant_data.get('model_name', '未知')}

【新聞情緒】
- FinBERT 分數：{news_data.get('sentiment_score', 0)}（-1 偏空 ↔ +1 偏多）
- 正面新聞：{pos_headlines if pos_headlines else ['近期無顯著正面新聞']}
- 負面新聞：{neg_headlines if neg_headlines else ['近期無顯著負面干擾']}
{news_status_prompt}

【投資人設定】
- 風格：{investor_profile.get('style', '穩健')} / 風險承受：{investor_profile.get('risk_tolerance', 'medium')}
- 最大可接受虧損：{investor_profile.get('max_loss_pct', 10)}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【你的回覆準則】

請先理解使用者真正想解決的投資問題，再選擇合適的回覆深度：
- 若是首次分析或問整體走勢，輸出完整三段報告。
- 若是針對已分析標的的具體追問，直接回答問題，3~6 句即可，不要重寫完整報告。
- 若是概念或指標解釋，先用白話說明，再連回當前數據。
- 若是買賣決策確認，給出有立場的條件式建議。

絕對禁止提到你如何分類問題，也禁止輸出任何內部分類字眼。

**完整報告格式：**

### 📌 一句話診斷
[用一句話說明這檔股票現在的投資位階與核心結論，要有態度，不要廢話]

#### 1️⃣ 市場情緒（新聞面）
[根據 FinBERT 分數解讀市場熱度，引用 1~2 則最具代表性的新聞標題說明為何如此。若分數為中立，說明市場觀望的原因]

#### 2️⃣ 量化模型（技術面）
[解讀量化訊號、預期報酬、市場狀態這三個數字的組合意義。例如：訊號買入但市場狀態偏弱，代表什麼？請用使用者聽得懂的語言說明]

#### 3️⃣ 你該怎麼做
[結合上述兩面，給出與使用者風格相符的具體操作建議。要回答「現在適合進場嗎？」「若已持有，要加碼、持有還是減碼？」「防守線/停損點建議在哪裡（依最大可接受虧損 {investor_profile.get('max_loss_pct', 10)}%）？」]

> ⚠️ 本報告基於量化模型與新聞情緒模型即時運算，僅供參考，不構成投資建議。

---
注意事項：
- 若情緒與量化訊號互相矛盾，必須明確說明哪個更值得信賴，以及理由
- 使用者問具體追問、概念解釋或決策確認時，不要輸出完整三段報告
- 禁止在量化數值解釋中說「受新聞情緒驅動」，量化模型純粹基於歷史價量
- 若使用者提到具體技術指標（如 RSI、MACD），必須結合當前數值解釋，若無數值請說明
- 語氣像一位有立場的朋友，不像一個免責的機器人
- 禁止輸出任何 HTML 標籤（例如 <br>、<table>、<div>）。只用 Markdown（換行請用空行或 Markdown 語法）
- 若使用者問題明確提到「本益比 / PER / PE / P/E / 估值偏高」，你必須直接回答「現在值不值得買」：
  1) 給出有立場的結論（買/等/不追）
  2) 說明高估值在何種策略下仍可接受（趨勢/動能 vs 價值）
  3) 若沒有即時本益比數字，禁止編造數字，改用使用者前提「本益比偏高」來推論風險與進場條件
'''

    yield from cached_generate_text_stream("investment_advice", prompt, ttl_seconds=43200)


    
    news_status_prompt = ""

def generate_portfolio_analysis_stream(rows, user_input, investor_profile, intent_type="PORTFOLIO"):
    investor_profile = investor_profile or {}

    portfolio_data_str = ""
    for r in rows:
        portfolio_data_str += (
            f"- {r['標的']}：訊號={r['量化訊號']} | 預期報酬={r['預期報酬率']} | "
            f"情緒={r['情緒分數']} | 近一年回撤={r['近一年最大回撤']}\n"
        )

    context = "比較多檔股票的優劣" if str(intent_type).upper() == "COMPARE" else "評估目前持股或投資組合健康度"

    prompt = f'''你是一位具備量化背景的台股投資顧問，能說人話、有立場、不說廢話。
請全程使用繁體中文。

【使用者問題】：{user_input}
【任務情境】：{context}

【各標的量化數據】：
{portfolio_data_str}
【投資人設定】：
- 風格：{investor_profile.get('style', '穩健')} / 最大可接受虧損：{investor_profile.get('max_loss_pct', 10)}%

━━━ 回覆要求 ━━━

**第一步：理解使用者真正的問題**
仔細看【使用者問題】，判斷他的核心需求是以下哪種，並據此調整回答重點：
- 「誰比較好？哪個 CP 值高？」→ 聚焦排名與對比
- 「我該怎麼配置？」→ 聚焦資金分配與比例
- 「誰風險最高？該先砍哪個？」→ 聚焦風險排序與停損
- 「這些訊號是什麼意思？」→ 聚焦解釋模型邏輯
- 「大盤崩了怎辦？」→ 聚焦壓力情境模擬
- 其他問題 → 直接回答，不要套用固定格式

**第二步：輸出回覆**

必須包含：
1. **一句話結論**：直接說出最重要的結論（誰最值得買/留/砍），不要模稜兩可
2. **交叉比較**：不要逐一列每檔的數據，要以「比較維度」來說明差異（報酬/風險對比、情緒熱度對比）
3. **具體行動建議**：結合投資人風格，給出可執行的操作指令，包含資金比例、進場方式、停損設定
   - 停損參考：以最大可接受虧損 {investor_profile.get('max_loss_pct', 10)}% 為上限

⚠️ 禁止說「視市場情況調整」「需要進一步觀察」等無實質內容的話。
⚠️ 每一檔標的都必須被提到至少一次。
⚠️ 若情緒與量化訊號矛盾，明確說明哪個指標在此情境下更可信。
⚠️ 只能根據上方【各標的量化數據】回答，禁止把名稱相似的股票視為同一家公司。
⚠️ 如果某一檔出現「抓取失敗 / 無法預測 / 無資料」，必須明講該檔目前資料不足，不可硬做排名。

> ⚠️ 本分析基於量化模型與新聞情緒模型，僅供參考，不構成投資建議。
⚠️ 禁止輸出任何 HTML 標籤（例如 <br>、<table>、<div>）。只用 Markdown（換行請用空行或 Markdown 語法）
'''

    yield from cached_generate_text_stream("portfolio_analysis", prompt, ttl_seconds=43200)


def generate_follow_up_answer_stream(user_input, dashboard_data, chat_history, user_news_snippet: str | None = None):
    dashboard_data = dashboard_data or {}
    if dashboard_data.get("mode") == "compare":
        rows = dashboard_data.get("compare_rows") or []
        stocks = dashboard_data.get("stocks") or []
        recommendation = dashboard_data.get("recommendation") or ""
        investor_profile = dashboard_data.get("investor_profile") or {}
        compared_names = dashboard_data.get("compared_names") or "、".join(
            f"{item.get('company_name') or item.get('ticker')}（{item.get('ticker')}）"
            for item in stocks
            if isinstance(item, dict)
        )
        table_lines = []
        if rows:
            columns = list(rows[0].keys())
            table_lines.append("| " + " | ".join(columns) + " |")
            table_lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
            for row in rows:
                table_lines.append("| " + " | ".join(str(row.get(col, "無")) for col in columns) + " |")
        compare_table = "\n".join(table_lines) if table_lines else "（比較表暫無可用資料）"

        recent_history = ""
        if chat_history:
            recent = [m for m in chat_history[-8:] if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
            recent_history = "\n".join(f"{m['role'].upper()}: {m['content'][:300]}" for m in recent)

        prompt = f'''你是一位台股投資顧問，正在延續上一輪「多檔比較」對話。
請全程使用繁體中文，直接回答使用者的追問，不要重新要求股票代號。

【使用者最新問題】：{user_input}

【上一輪比較標的】：
{compared_names}

【上一輪比較表】：
{compare_table}

【上一輪建議】：
{recommendation if recommendation else "（上一輪沒有額外建議文字）"}

【對話歷史】：
{recent_history if recent_history else "（無歷史）"}

【投資設定】：
- 風格：{investor_profile.get('style', '穩健')}
- 最大可接受虧損：{investor_profile.get('max_loss_pct', 10)}%

━━━ 回覆準則 ━━━
- 這是比較結果的追問，必須沿用上方比較表與建議回答。
- 如果使用者問「只能選一檔 / CP 值 / 預期報酬 vs 風險 / 如何配置」，請明確給出首選、次選、觀望或避免，以及理由。
- 若某檔資料不足，明講該檔資料不足，但仍用其他已有欄位做保守判斷。
- 不要說「查無標的」或要求重新輸入代號，除非使用者明確提到不在上一輪比較表中的新股票。
- 禁止說明你如何判斷問題，也禁止輸出任何內部分類字眼。
- 禁止輸出任何 HTML 標籤（例如 <br>、<table>、<div>）。只用 Markdown。
'''
        yield from cached_generate_text_stream("compare_follow_up", prompt, ttl_seconds=43200)
        return

    ticker = dashboard_data.get("ticker", "未知")
    company_name = dashboard_data.get("company_name", ticker)
    stock_data = dashboard_data.get("stock_data") or {}
    quant_data = dashboard_data.get("quant_data") or {}
    news_data = dashboard_data.get("news_data") or {}
    investor_profile = dashboard_data.get("investor_profile") or {}
    raw_news = news_data.get("raw_news") or []
    evidence_headlines = [n.get("新聞標題", "") for n in raw_news[:4] if n.get("新聞標題")]

    recent_history = ""
    if chat_history:
        recent = [m for m in chat_history[-8:] if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
        recent_history = "\n".join(f"{m['role'].upper()}: {m['content'][:300]}" for m in recent)

    snippet_block = ""
    if user_news_snippet:
        snippet_block = f"\n\n【使用者提供的新聞片段（僅作追問參考）】\n{user_news_snippet.strip()[:500]}"

    prompt = f'''你是一位台股投資顧問，正在與使用者針對「{company_name}（{ticker}）」進行持續對話。
請全程使用繁體中文，語氣像一位有立場的朋友而非免責機器人。

【使用者最新問題】：{user_input}
{snippet_block}

【對話歷史】：
{recent_history if recent_history else "（無歷史）"}

【可用的分析數據】：
- 目前股價：{stock_data.get('latest_price', '未知')}
- 量化訊號：{quant_data.get('signal', '未知')}
- 預期報酬：{round(quant_data.get('predicted_return', 0) * 100, 3)}%
- 市場狀態：{quant_data.get('regime', '未知')}
- 近一年最大回撤：{quant_data.get('max_dd', 'N/A')}
- 情緒分數：{news_data.get('sentiment_score', 0)}（-1 偏空 ↔ +1 偏多）
- 新聞佐證：{evidence_headlines if evidence_headlines else ['無近期新聞']}
- 投資風格：{investor_profile.get('style', '穩健')} / 最大可接受虧損：{investor_profile.get('max_loss_pct', 10)}%

━━━ 回覆準則 ━━━

根據使用者真正想解決的投資問題，選擇合適的回覆深度：

**操作型問題**（停損怎設？該加碼嗎？何時進場？）
→ 給出有數字、有條件的具體建議。例如：「目前回撤 X%，以你的最大虧損 Y% 來看，建議停損設在 Z 元」

**解釋型問題**（這個訊號是什麼意思？Regime 怎麼看？）
→ 先白話說明，加一個生活比喻，再結合當前數值解讀

**情緒/心態型問題**（我很擔心、這樣正常嗎？）
→ 先給予認可，再用數據說明實際情況，提供理性視角

**策略型問題**（要不要換股？要分批還是單筆？）
→ 給出有條件的建議（「如果 A，則 B；如果 C，則 D」），不要只說「視情況而定」

**注意**：
- 這是追問，**絕對不要重寫完整分析報告**
- 禁止說明你如何判斷問題，也禁止輸出任何內部分類字眼。
- 長度依問題複雜度調整：簡單問題 2~3 句，複雜問題可到 6~8 句
- 若問題涉及使用者沒問到的面向，不要主動展開，聚焦在他的問題上
- 結尾加一句簡短風險提醒即可
 - 禁止輸出任何 HTML 標籤（例如 <br>、<table>、<div>）。只用 Markdown（換行請用空行或 Markdown 語法）
'''
