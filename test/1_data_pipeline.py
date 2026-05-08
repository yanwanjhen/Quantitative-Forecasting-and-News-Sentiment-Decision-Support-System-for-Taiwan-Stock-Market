import time
import re
import difflib
import json
import urllib.parse
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timedelta
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai

# ==========================================
# 0. 參數設定
# ==========================================
GEMINI_API_KEY = "AIzaSyCJlKuXsY6cQzWyZwytgIkoFP3chiSvNAQ" # ⚠️請替換為您的金鑰
genai.configure(api_key=GEMINI_API_KEY)
model_llm = genai.GenerativeModel('gemini-2.5-flash')

TARGET_COMPANY = "台積電"

# 關鍵字字典
positive_words = ["大漲", "上揚", "走揚", "創高", "反彈", "翻倍", "新高", "收紅", "填息", "帶量", "噴發", "站穩", "續強", "轉強", "漲停", "站回", "收復", "再漲", "衝上", "漲逾", "強彈", "衝高", "穩守", "撐腰", "紅燈", "亮紅燈", "漲", "領航", "強勢", "有戲", "強升", "登高", "點火", "回溫", "暴衝", "飆", "強攻", "看俏", "創新高", "飆漲", "成長", "利多", "看好", "突破", "獲利", "增長", "上修", "升溫", "強勁", "優於", "增加", "提升", "回升", "轉盈", "亮眼", "達標", "滿載", "營收", "創紀錄", "受惠", "攜手", "結盟", "年增", "月增","擴增", "強化","私募", "入股", "大咖", "綁樁", "震撼彈", "缺貨", "缺爆", "搶單", "奪下", "通吃", "大單", "急單", "護盤", "雙成長", "添柴", "燒", "佳", "旺", "優", "報喜", "繳出", "每股賺", "賺", "樂觀", "積極", "正面", "買超", "加碼", "敲進", "回補", "進駐", "連買", "買單", "掃貨", "大買", "回流", "注資", "抄底", "力挺", "狂湧", "青睞", "照顧"]
negative_words = ["下跌", "大跌", "重挫", "暴跌", "下滑", "腰斬", "新低", "走低", "探底", "貼息", "收黑", "摜破", "失守", "承壓", "遇阻", "跌穿", "跌停", "回檔", "修正", "疲軟", "震盪", "翻黑", "得而復失", "熄火", "平盤", "跌", "綠燈", "亮綠燈", "崩", "殺", "吞跌停", "下探", "未站穩", "崩跌", "摔", "狂震", "慘墜", "陣亡", "探低", "破底", "重摔", "利空", "衰退", "看淡", "虧損", "下修", "降溫", "疲弱", "不如", "減少", "下降", "轉虧", "慘淡", "年減", "月減", "不如預期", "蒸發", "縮水", "砍單", "瓶頸", "示警", "衝擊", "危機", "重創", "遭殃", "掉單", "流失", "保守", "裁員", "違約", "調查", "訴訟", "停滯", "失利", "堪憂", "降評", "挨刀", "每股虧", "虧", "差", "弱", "損", "悲觀", "消極", "負面", "賣壓", "賣超", "減持", "出脫", "拋售", "撤資", "連賣", "倒貨", "提款", "瘋狂拋售", "棄守", "調節", "狂砍"]
ambiguous_positive_words = {"攜手", "突破", "優", "賺", "飆", "大咖", "震撼彈", "照顧", "佳", "旺", "燒"}
finance_context_terms = {"營收", "獲利", "財報", "法說", "訂單", "供應鏈", "股價", "個股", "股市", "股票", "股東", "台股", "EPS", "殖利率", "本益比", "投資", "配息", "除息", "買超", "賣超", "市值", "大盤", "外資", "法人"}
non_investment_context_terms = {"基金會", "文教", "職缺", "員工", "市長", "城市", "浪漫", "健康", "候選", "永續", "公益", "校園", "徵才", "招募", "表白", "董事", "活動", "供應商"}
negative_keyword_exclusions = {"下調": ("下調漲",)}

# 載入 FinBERT 模型
print("🧠 正在載入 yiyanghkust/finbert-tone-chinese 模型...")
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone-chinese")
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone-chinese")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finbert_model.to(device)
finbert_model.eval()

# ==========================================
# 1. 抓取 2026 年 2 月至 4 月的新聞
# ==========================================
def get_google_news_feb_to_apr(keyword):
    print(f"📰 開始抓取【{keyword}】 2026年 2月至 4月 的新聞...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    all_titles = []
    
    # 設定時間區間
    start_date = datetime(2026, 2, 1)
    end_date = datetime(2026, 4, 30)
    current_date = start_date
    
    # 每 5 天為一區間抓取，避免遺漏
    while current_date < end_date:
        next_date = min(current_date + timedelta(days=5), end_date)
        query_str = f'"{keyword}" -site:cmoney.tw -同學會 -討論 -PTT -Dcard'
        query = urllib.parse.quote(query_str)
        url = f"https://news.google.com/rss/search?q={query}+after:{current_date.strftime('%Y-%m-%d')}+before:{next_date.strftime('%Y-%m-%d')}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        
        try:
            res = requests.get(url, headers=headers, timeout=10)
            root = ET.fromstring(res.text)
            titles = [item.find('title').text for item in root.findall('.//item') if item.find('title') is not None]
            all_titles.extend(titles)
            time.sleep(1) 
        except Exception as e:
            print(f"抓取發生錯誤: {e}")
        current_date = next_date
            
    return list(set(all_titles))

def clean_and_dedup(titles):
    deduped_titles = []
    for raw_title in titles:
        clean_text = re.sub(r'^.*?》|《.*?》|【.*?】|〈.*?〉|「.*?」|『.*?』|快訊／|- 上市櫃', '', raw_title.rsplit(' - ', 1)[0]).strip()
        if len(clean_text) < 7: continue
        is_dup = any(difflib.SequenceMatcher(None, clean_text, re.sub(r'^.*?》|《.*?》|【.*?】|〈.*?〉|「.*?」|『.*?』|快訊／|- 上市櫃', '', seen.rsplit(' - ', 1)[0]).strip()).ratio() > 0.35 or clean_text[:5] == re.sub(r'^.*?》|《.*?》|【.*?】|〈.*?〉|「.*?」|『.*?』|快訊／|- 上市櫃', '', seen.rsplit(' - ', 1)[0]).strip()[:5] for seen in deduped_titles)
        if not is_dup: deduped_titles.append(raw_title)
    return deduped_titles

def filter_with_llm(company_name, news_list):
    print(f"🤖 呼叫 Gemini 進行審查 (共 {len(news_list)} 筆)...")
    if not news_list: return [], []
    
    keep_list, drop_list = [], []
    batch_size = 30
    
    for i in range(0, len(news_list), batch_size):
        batch = news_list[i:i+batch_size]
        news_text = "\n".join([f"[{j}] {title}" for j, title in enumerate(batch)])
        prompt = f"""請審查新聞是否保留以分析【{company_name}】的獨立情緒。
        1. 若未以公司為主角、無直接關聯，丟棄(drop)。
        2. 若同時出現多檔個股且非聚焦目標，丟棄(drop)。
        嚴格輸出 JSON: [ {{"original_title": "...", "action": "keep或drop", "reason": "..."}} ]
        新聞：{news_text}"""
        try:
            response = model_llm.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            results = json.loads(response.text)
            for item in results:
                if item.get("action") == "keep": keep_list.append(item["original_title"])
                else: drop_list.append(item["original_title"])
        except Exception as e:
            print(f"批次處理錯誤: {e}")
            keep_list.extend([t for t in batch if company_name in t])
        time.sleep(2)
    return keep_list, drop_list


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
        if any(phrase in text for phrase in negative_keyword_exclusions.get(word, ())):
            continue
        neg_hits.append(word)
    return pos_hits, neg_hits

def calculate_scores(text, target_company):
    clean_title = re.sub(r'^.*?》|《.*?》|【.*?】|〈.*?〉|「.*?」|『.*?』|快訊／|- 上市櫃', '', text).strip()
    segments = re.split(r'[，,。：:；;！!？?、\s]+', clean_title)
    relevant_segments = [seg for seg in segments if target_company in seg]
    target_text = " ".join(relevant_segments) if relevant_segments else clean_title

    inputs = tokenizer(target_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad(): logits = finbert_model(**inputs).logits[0]
        
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # [Baseline]: LLM-filtered FinBERT (純模型機率)
    base_idx = torch.argmax(probs).item()
    base_label = "Positive" if base_idx == 1 else "Negative" if base_idx == 2 else "Neutral"
    
    # [Proposed]: LLM-filtered Hybrid FinBERT (混合權重)
    l_neu, l_pos, l_neg = logits[0].item(), logits[1].item(), logits[2].item()
    p_neu = probs[0].item()
    base_score = torch.tanh(torch.tensor((l_pos - l_neg) / 1.5)).item()
    
    pos_hits, neg_hits = _keyword_hits(target_text)
    keyword_score = torch.tanh(torch.tensor((len(pos_hits) - len(neg_hits) * 1.2) / 2.0)).item()

    w_model, w_keyword = (0.3, 0.7) if p_neu > 0.5 else (0.8, 0.2)
    final_score = (base_score * w_model) + (keyword_score * w_keyword)
    if abs(keyword_score) > 0.7 and abs(final_score) < 0.3: final_score = (final_score + keyword_score) / 2
    if _is_non_investment_context(target_text) and not _is_finance_context(target_text): final_score *= 0.5
    final_score = max(min(final_score, 1.0), -1.0)
    
    prop_label = "Positive" if final_score > 0.15 else "Negative" if final_score < -0.15 else "Neutral"
    return base_label, prop_label, round(final_score, 4)

# ==========================================
# 執行
# ==========================================
if __name__ == "__main__":
    raw_titles = get_google_news_feb_to_apr(TARGET_COMPANY)
    deduped_titles = clean_and_dedup(raw_titles)
    keep_list, drop_list = filter_with_llm(TARGET_COMPANY, deduped_titles)
    
    # 解決 Gap 2 (LLM Filtering Precision)：自動抽取 50 筆被丟棄的新聞給您審查
    df_drop = pd.DataFrame({"Dropped_News_Title": drop_list})
    sample_drop = df_drop.sample(n=min(50, len(df_drop)), random_state=42)
    sample_drop['Is_Truly_Irrelevant_or_Noise_(Y/N)'] = '' # 讓您手動填 Y 或 N
    sample_drop.to_csv("llm_dropped_sample_50.csv", index=False, encoding="utf-8-sig")
    print(f"✅ 已輸出 50 筆被 LLM 丟棄的新聞至 'llm_dropped_sample_50.csv' 供您人工驗證。")
    
    # 處理保留的新聞並評分
    results = []
    for title in keep_list:
        base_label, prop_label, prop_score = calculate_scores(title, TARGET_COMPANY)
        results.append({
            "News_Title": title,
            "LLM_filtered_FinBERT_Label": base_label,    # 這是傳統 FinBERT
            "LLM_filtered_Hybrid_Label": prop_label,     # 這是您的機制
            "Proposed_Score": prop_score
        })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("experiment_results_2026.csv", index=False, encoding="utf-8-sig")
    print("✅ 分析完成！資料已儲存至 'experiment_results_2026.csv'")
