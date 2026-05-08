try:
    import streamlit as st
except Exception:
    st = None
import torch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. 載入專業中文金融 FinBERT 模型 (對齊原始標註檔案)
# ==========================================
_cache_resource = st.cache_resource if st is not None else (lambda func: func)


@_cache_resource
def load_finbert_model():
    model_name = "yiyanghkust/finbert-tone-chinese"
    print(f"🧠 正在載入專業金融 FinBERT 模型 ({model_name}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        # 在這個模型上，Apple MPS 的分數和 CPU 會出現可觀漂移，
        # 會直接影響人工標註驗證結果，因此預設優先使用 CPU 以維持可重現性。
        device = torch.device("cpu")
        
    model.to(device)
    return tokenizer, model, device

# ==========================================
# 🚀 擴展情緒詞典與關鍵字萃取 (結合 CSV 台積電與鴻海近期重點字)
# ==========================================
positive_words = [
    # 📈 價格與技術面
    "大漲", "上揚", "走揚", "創高", "反彈", "翻倍", "新高", "收紅", "填息", "帶量", 
    "噴發", "站穩", "續強", "轉強", "漲停", "站回", "收復",
    "再漲", "衝上", "漲逾", "強彈", "衝高", "穩守", "撐腰", "紅燈", "亮紅燈", "漲",
    "領航", "強勢", "有戲", "強升", "登高", "點火", "回溫", "暴衝", "飆", "強攻", "看俏",
    "創新高", "飆漲", # 新增
    
    # 🏢 基本面與營運 / 📌 企業重大事件 (Event-Driven)
    "成長", "利多", "看好", "突破", "獲利", "增長", "上修", "升溫", "強勁", 
    "優於", "增加", "提升", "回升", "轉盈", "亮眼", "達標", "滿載", "營收", 
    "創紀錄", "受惠", "攜手", "結盟", "年增", "月增", 
    "擴增", "強化", # 產能與發展擴充
    "私募", "入股", "大咖", "綁樁", "震撼彈", "缺貨", "缺爆", # 事件驅動與產能利多
    "搶單", "奪下", "通吃", "大單", "急單", "護盤", # 訂單面
    "雙成長", "添柴", "燒", "佳", "旺", "優", "報喜", "繳出", "每股賺", "賺", # 營收與話題 (新增)
    
    # 💰 籌碼面與市場情緒
    "樂觀", "積極", "正面", "買超", "加碼", "敲進", "回補", "進駐", "連買", 
    "買單", "掃貨", "大買", "回流", "注資", "抄底", "力挺", "狂湧", "青睞", "照顧"
]

negative_words = [
    # 📉 價格與技術面
    "下跌", "大跌", "重挫", "暴跌", "下滑", "腰斬", "新低", "走低", "探底", "貼息", 
    "收黑", "摜破", "失守", "承壓", "遇阻", "跌穿", "跌停", "回檔", "修正", "疲軟",
    "震盪", "翻黑", "得而復失", "熄火", "平盤", "跌", "綠燈", "亮綠燈", "崩", "殺",
    "吞跌停", "下探", "未站穩", "崩跌", "摔", "狂震", "慘墜", "陣亡", "空頭", "賣壓", # 技術面與價格
    "探低", "破底", "重摔", "跳水", "急跌", "殺尾盤", "破線", "走弱", "弱勢", # 新增
    
    # 🏢 基本面與營運
    "利空", "衰退", "看淡", "虧損", "下修", "降溫", "疲弱", "不如", "減少", 
    "下降", "轉虧", "慘淡", "年減", "月減", "不如預期", "蒸發", "縮水", "砍單",
    "瓶頸", "示警", "衝擊", "危機", "重創", "遭殃", "惡化", "拖累", "警訊", # 產能與國際局勢擴充
    "掉單", "流失", "保守", "裁員", "違約", "調查", "訴訟", "延遲", "延期", # 負面事件
    "停滯", "失利", "堪憂", "降評", "挨刀", "看壞", "下調", "不佳", # 營運展望
    "每股虧", "虧", "差", "弱", "損", "營收減", "衰", # 新增
    
    # 💰 籌碼面與市場情緒
    "悲觀", "消極", "負面", "賣壓", "賣超", "減持", "出脫", "拋售", "撤資", 
    "連賣", "倒貨", "提款", "瘋狂拋售", "棄守", "調節", "狂砍", "逃命", "甩賣", "大逃殺"
]

# 這些詞若沒有財經脈絡，很容易把 CSR / 徵才 / 活動新聞誤判成正向投資情緒。
ambiguous_positive_words = {
    "攜手", "突破", "優", "賺", "飆", "大咖", "震撼彈", "照顧", "佳", "旺", "燒"
}

finance_context_terms = {
    "營收", "獲利", "財報", "法說", "訂單", "供應鏈", "股價", "個股", "股市",
    "股票", "股東", "台股", "EPS", "殖利率", "本益比", "投資", "配息", "除息",
    "買超", "賣超", "市值", "大盤", "外資", "法人"
}

non_investment_context_terms = {
    "基金會", "文教", "職缺", "員工", "市長", "城市", "浪漫", "健康", "候選",
    "永續", "公益", "校園", "徵才", "招募", "表白", "董事", "活動", "供應商"
}

negative_keyword_exclusions = {
    # 「以下調漲」描述的是價格上調，不能被「下調」誤判成利空。
    "下調": ("下調漲",),
}


def _contains_any(text, terms):
    return any(term in text for term in terms)


def _is_finance_context(text):
    return _contains_any(text, finance_context_terms)


def _is_non_investment_context(text):
    return _contains_any(text, non_investment_context_terms)


def _keyword_hits(text):
    has_finance_context = _is_finance_context(text)
    pos_hits = []
    for word in positive_words:
        if word not in text:
            continue
        if word in ambiguous_positive_words and not has_finance_context:
            continue
        pos_hits.append(word)

    neg_hits = []
    for word in negative_words:
        if word not in text:
            continue
        blocked = any(phrase in text for phrase in negative_keyword_exclusions.get(word, ()))
        if blocked:
            continue
        neg_hits.append(word)
    return pos_hits, neg_hits

def extract_keywords(text):
    if pd.isna(text) or not isinstance(text, str):
        return "", ""
    text = str(text)
    hits_pos, hits_neg = _keyword_hits(text)
    return "、".join(hits_pos), "、".join(hits_neg)

# ==========================================
# 2. 情緒連續分數計算 
# ==========================================
def get_finbert_continuous_score(text, target_company=None):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return 0.0

    tokenizer, finbert_model, device = load_finbert_model()

    # ==========================================================
    # 🎯 標的專屬情緒鎖定 (Target-Specific Sentiment Focus)
    # ==========================================================
    # 如果指定了公司名稱，且不是大盤/台股，嘗試切割句子只分析相關段落
    if target_company and target_company not in ["大盤", "台股", "加權指數"]:
        # 使用常見的分句符號切割 (逗號、頓號、空格等)
        segments = re.split(r'[，,。：:；;！!？?、\s]+', text)
        
        # 找出包含目標公司的段落
        # 這裡做一個簡單的優化：如果公司名是兩字以上，嘗試匹配
        relevant_segments = [seg for seg in segments if target_company in seg]
        
        # 如果找到了專屬於該公司的段落，就只分析這一段
        if len(relevant_segments) > 0:
            text = " ".join(relevant_segments)

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    ).to(device)

    with torch.no_grad():
        outputs = finbert_model(**inputs)
        
    # --------------------------------------------------------
    # 🚀 改進版算法：使用 Logits (原始數值) 取代機率，解決中性分數過高導致數值太小的問題
    # --------------------------------------------------------
    logits = outputs.logits[0] # 取出原始 logits
    
    if len(logits) == 3:
        # 根據 yiyanghkust/finbert-tone-chinese，通常順序為 [Neutral, Positive, Negative]
        # 但原始代碼假設是 [Neu, Pos, Neg]，我們這裡保持一致的 Logit 取法
        l_neu = logits[0].item()
        l_pos = logits[1].item()
        l_neg = logits[2].item()
    elif len(logits) == 2:
        l_neg = logits[0].item()
        l_pos = logits[1].item()
        l_neu = 0.0
    else:
        l_pos, l_neg, l_neu = 0.0, 0.0, 0.0
        
    # 核心算法變更：
    # 1. 計算正負向的「相對強度」 (忽略中性分數的絕對值)
    # 2. 如果正負向 Logit 差距很小，代表真的很中性；如果差距大，代表有明確觀點
    raw_score = l_pos - l_neg
    
    # 3. 使用 Scaling 係數來放大訊號 (Logit 差 1.5 分左右就視為顯著)
    scale_factor = 1.5 
    
    # 4. 透過 Tanh 函數將分數映射到 -1 ~ 1 之間 (平滑過渡)
    #    例如: 差值 0 -> 0; 差值 1.5 -> 0.76; 差值 -2 -> -0.96
    base_score = torch.tanh(torch.tensor(raw_score / scale_factor)).item()
    
    # 計算機率僅用於輔助判斷 (如 Neutral 是否極端高)
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    if len(probabilities) == 3: p_neu = probabilities[0].item()
    else: p_neu = 0.0

    # ==========================================================
    # ⚖️ 混合權重修正機制 (Hybrid Weighted Adjustment)
    # ==========================================================
    # 計算關鍵字指標：統計正負面詞彙數量
    pos_hits, neg_hits = _keyword_hits(text)
    
    # 1. 關鍵字分數 (Keyword Score): 使用 Tanh 映射，讓 2-3 個關鍵字就能達到強烈分數
    #    除以 2.0 代表每多 2 個淨關鍵字，分數會往極端靠近; 負面字權重 1.2 倍
    keyword_score = torch.tanh(torch.tensor((len(pos_hits) - len(neg_hits) * 1.2) / 2.0)).item() 

    # 2. 動態權重 (Dynamic Weighting): 依據模型對「中性」的確信度來決定聽誰的
    if p_neu > 0.5:
        # Case A: 模型認為是中性 (Neutral) -> 「關鍵字」權重調高
        # 這解決了 "利多新聞被模型判為中性" 的常見問題
        w_model = 0.3
        w_keyword = 0.7
    else:
        # Case B: 模型已有明確多空看法 (Pos/Neg) -> 「模型」權重調高
        # 關鍵字僅作輔助，避免誤導
        w_model = 0.8
        w_keyword = 0.2
        
    # 3. 融合計算
    final_score = (base_score * w_model) + (keyword_score * w_keyword)

    # 4. 極端防護 (Safety Net): 若關鍵字訊號極強 (如: 崩盤、漲停)，但合併分數卻中庸
    #    強制拉向關鍵字方向，避免被模型的中性拉回太深
    if abs(keyword_score) > 0.7 and abs(final_score) < 0.3:
         final_score = (final_score + keyword_score) / 2

    # 對 CSR / 徵才 / 活動等非投資題材降權，避免「公司有被提到」就被誤解成利多。
    if _is_non_investment_context(text) and not _is_finance_context(text):
        final_score *= 0.5

    final_score = max(min(final_score, 1.0), -1.0)
    return round(final_score, 4)
