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
# 0. 參數設定 (請填入您的 API Key)
# ==========================================
GEMINI_API_KEY = "AIzaSyCJlKuXsY6cQzWyZwytgIkoFP3chiSvNAQ" # 替換為您的金鑰
genai.configure(api_key=GEMINI_API_KEY)
model_llm = genai.GenerativeModel('gemini-2.5-flash')

TARGET_COMPANY = "台積電" # 測試標的
TEST_DAYS = 30 # 測試天數 (模擬近一個月)

# 關鍵字字典 (從您原本的程式碼擷取)
positive_words = ["大漲", "上揚", "走揚", "創高", "反彈", "翻倍", "新高", "收紅", "填息", "帶量", 
    "噴發", "站穩", "續強", "轉強", "漲停", "站回", "收復", "再漲", "衝上", "漲逾", "強彈", "衝高", "穩守", "撐腰", "紅燈", "亮紅燈", "漲", "領航", "強勢", "有戲", "強升", "登高", "點火", "回溫", "暴衝", "飆", "強攻", "看俏", "創新高", "飆漲", "成長", "利多", "看好", "突破", "獲利", "增長", "上修", "升溫", "強勁", "優於", "增加", "提升", "回升", "轉盈", "亮眼", "達標", "滿載", "營收", "創紀錄", "受惠", "攜手", "結盟", "年增", "月增","擴增", "強化","私募", "入股", "大咖", "綁樁", "震撼彈", "缺貨", "缺爆", "搶單", "奪下", "通吃", "大單", "急單", "護盤", "雙成長", "添柴", "燒", "佳", "旺", "優", "報喜", "繳出", "每股賺", "賺", "樂觀", "積極", "正面", "買超", "加碼", "敲進", "回補", "進駐", "連買", "買單", "掃貨", "大買", "回流", "注資", "抄底", "力挺", "狂湧", "青睞", "照顧"]
negative_words = ["下跌", "大跌", "重挫", "暴跌", "下滑", "腰斬", "新低", "走低", "探底", "貼息", 
    "收黑", "摜破", "失守", "承壓", "遇阻", "跌穿", "跌停", "回檔", "修正", "疲軟", "震盪", "翻黑", "得而復失", "熄火", "平盤", "跌", "綠燈", "亮綠燈", "崩", "殺",
    "吞跌停", "下探", "未站穩", "崩跌", "摔", "狂震", "慘墜", "陣亡", 
    "探低", "破底", "重摔", "利空", "衰退", "看淡", "虧損", "下修", "降溫", "疲弱", "不如", "減少", 
    "下降", "轉虧", "慘淡", "年減", "月減", "不如預期", "蒸發", "縮水", "砍單",
    "瓶頸", "示警", "衝擊", "危機", "重創", "遭殃", # 產能與國際局勢擴充
    "掉單", "流失", "保守", "裁員", "違約", "調查", "訴訟", # 負面事件
    "停滯", "失利", "堪憂", "降評", "挨刀", # 營運展望
    "每股虧", "虧", "差", "弱", "損", "悲觀", "消極", "負面", "賣壓", "賣超", "減持", "出脫", "拋售", "撤資", 
    "連賣", "倒貨", "提款", "瘋狂拋售", "棄守", "調節", "狂砍"]

# ==========================================
# 1. 載入 FinBERT 模型
# ==========================================
print("🧠 正在載入 yiyanghkust/finbert-tone-chinese 模型...")
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone-chinese")
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone-chinese")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finbert_model.to(device)
finbert_model.eval()

# ==========================================
# 2. 定義各階段函數
# ==========================================

def get_google_news_history(keyword, days=30):
    """模擬過去 N 天的新聞抓取 (分段抓取避免 RSS 漏資料)"""
    print(f"📰 開始抓取【{keyword}】過去 {days} 天的新聞...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    all_titles = []
    
    # 為了抓取更完整，我們以 5 天為一個區間去搜尋
    for i in range(0, days, 5):
        end_date = datetime.now() - timedelta(days=i)
        start_date = end_date - timedelta(days=5)
        
        query_str = f'"{keyword}" -site:cmoney.tw -同學會 -討論 -PTT -Dcard'
        query = urllib.parse.quote(query_str)
        url = f"https://news.google.com/rss/search?q={query}+after:{start_date.strftime('%Y-%m-%d')}+before:{end_date.strftime('%Y-%m-%d')}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        
        try:
            res = requests.get(url, headers=headers, timeout=10)
            root = ET.fromstring(res.text)
            titles = [item.find('title').text for item in root.findall('.//item') if item.find('title') is not None]
            all_titles.extend(titles)
            time.sleep(1) # 避免被 Google 阻擋
        except Exception as e:
            print(f"抓取區間發生錯誤: {e}")
            
    return list(set(all_titles)) # 初步排除完全重複的字串

def clean_and_dedup(titles):
    """標題清洗與相似度去重"""
    deduped_titles = []
    for raw_title in titles:
        core_title = raw_title.rsplit(' - ', 1)[0].strip()
        clean_text = re.sub(r'^.*?》|《.*?》|【.*?】|〈.*?〉|「.*?」|『.*?』|快訊／|- 上市櫃', '', core_title).strip()
        
        if len(clean_text) < 7: continue
            
        is_dup = False
        for seen in deduped_titles:
            seen_clean = re.sub(r'^.*?》|《.*?》|【.*?】|〈.*?〉|「.*?」|『.*?』|快訊／|- 上市櫃', '', seen.rsplit(' - ', 1)[0]).strip()
            similarity = difflib.SequenceMatcher(None, clean_text, seen_clean).ratio()
            if similarity > 0.35 or clean_text[:5] == seen_clean[:5]:
                is_dup = True
                break
        
        if not is_dup:
            deduped_titles.append(raw_title)
    return deduped_titles

def filter_with_llm(company_name, news_list):
    """LLM 實體關聯審查"""
    print(f"🤖 正在呼叫 Gemini 進行 LLM 實體關聯審查 (共 {len(news_list)} 筆)...")
    if not news_list: return []
    
    # 批次處理，避免 prompt 過長
    batch_size = 30
    refined_news = []
    
    for i in range(0, len(news_list), batch_size):
        batch = news_list[i:i+batch_size]
        news_text = "\n".join([f"[{j}] {title}" for j, title in enumerate(batch)])
        prompt = f"""
        請對以下新聞進行實體關聯審查，判斷是否保留該新聞。使用者目前只想分析【{company_name}】的「獨立情緒」。
        規則：
        1. 若標題未以 {company_name} 為主角或無直接關聯，丟棄 (drop)。
        2. 若同時出現多檔個股且非聚焦 {company_name}，丟棄 (drop)。
        3. 若保留 (keep)，必須 100% 輸出原始標題。
        請嚴格輸出 JSON Array: [ {{"original_title": "...", "action": "keep或drop", "reason": "..."}} ]
        
        新聞列表：
        {news_text}
        """
        try:
            response = model_llm.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            results = json.loads(response.text)
            refined_news.extend([item["original_title"] for item in results if item.get("action") == "keep"])
        except Exception as e:
            print(f"LLM 批次處理錯誤: {e}")
            # Fallback
            refined_news.extend([t for t in batch if company_name in t])
        time.sleep(2) # 避免 API rate limit
        
    return refined_news

def get_baseline_finbert(text):
    """對照組：傳統原始 FinBERT (單純取最高機率類別)"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = finbert_model(**inputs).logits[0]
    # yiyanghkust 順序通常為 [Neutral, Positive, Negative]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    class_idx = torch.argmax(probs).item()
    if class_idx == 1: return "Positive"
    elif class_idx == 2: return "Negative"
    else: return "Neutral"

def get_proposed_finbert_score(text, target_company):
    """實驗組：本研究提出的混合權重機制連續分數 (已拓寬中性門檻)"""
    clean_title = re.sub(r'^.*?》|《.*?》|【.*?】|〈.*?〉|「.*?」|『.*?』|快訊／|- 上市櫃', '', text).strip()
    
    # 專屬段落切割
    segments = re.split(r'[，,。：:；;！!？?、\s]+', clean_title)
    relevant_segments = [seg for seg in segments if target_company in seg]
    target_text = " ".join(relevant_segments) if relevant_segments else clean_title

    inputs = tokenizer(target_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = finbert_model(**inputs).logits[0]
        
    l_neu, l_pos, l_neg = logits[0].item(), logits[1].item(), logits[2].item()
    p_neu = torch.nn.functional.softmax(logits, dim=-1)[0].item()
    
    # 基礎分數
    raw_score = l_pos - l_neg
    base_score = torch.tanh(torch.tensor(raw_score / 1.5)).item()
    
    # 關鍵字分數
    pos_hits = sum(1 for w in positive_words if w in target_text)
    neg_hits = sum(1 for w in negative_words if w in target_text)
    keyword_score = torch.tanh(torch.tensor((pos_hits - neg_hits * 1.2) / 2.0)).item() 

    # 動態權重
    if p_neu > 0.5:
        w_model, w_keyword = 0.3, 0.7
    else:
        w_model, w_keyword = 0.8, 0.2
        
    final_score = (base_score * w_model) + (keyword_score * w_keyword)
    if abs(keyword_score) > 0.7 and abs(final_score) < 0.3:
         final_score = (final_score + keyword_score) / 2
            
    final_score = max(min(final_score, 1.0), -1.0)
    
    # ==========================================
    # 改善 1：拓寬「中性區間（Neutral Band）」的門檻
    # 原本設定為 ±0.05，現改為 ±0.15，以過濾微弱情緒雜訊
    # ==========================================
    threshold = 0.15 
    
    if final_score > threshold: 
        return "Positive", final_score
    elif final_score < -threshold: 
        return "Negative", final_score
    else: 
        return "Neutral", final_score

# ==========================================
# 3. 執行實驗與統計
# ==========================================
def run_experiment():
    print("🚀 實驗開始...")
    t_start = time.time()
    
    # Step 1: 抓取原始資料
    raw_titles = get_google_news_history(TARGET_COMPANY, TEST_DAYS)
    total_raw = len(raw_titles)
    daily_avg = total_raw // TEST_DAYS if TEST_DAYS > 0 else 0
    print(f"✅ 取得原始新聞: {total_raw} 筆 (平均每日 {daily_avg} 筆)")
    
    # Step 2: 去重處理
    deduped_titles = clean_and_dedup(raw_titles)
    total_dedup = len(deduped_titles)
    print(f"✅ 去重後保留: {total_dedup} 筆")
    
    # Step 3: LLM 實體審查
    llm_filtered_titles = filter_with_llm(TARGET_COMPANY, deduped_titles)
    total_llm = len(llm_filtered_titles)
    print(f"✅ LLM 實體過濾後保留: {total_llm} 筆的高關聯新聞")
    
    # Step 4: 評分對比
    results = []
    baseline_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    proposed_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    
    print("📊 正在進行模型對比運算...")
    for title in llm_filtered_titles:
        # 傳統基準
        base_label = get_baseline_finbert(title)
        baseline_counts[base_label] += 1
        
        # 我們的機制
        prop_label, prop_score = get_proposed_finbert_score(title, TARGET_COMPANY)
        proposed_counts[prop_label] += 1
        
        results.append({
            "News_Title": title,
            "Baseline_Label": base_label,
            "Proposed_Label": prop_label,
            "Proposed_Score": round(prop_score, 4)
        })
        
    # 計算百分比
    def calc_pct(counts):
        total = sum(counts.values())
        if total == 0: return 0, 0, 0
        return round(counts["Positive"]/total*100, 1), round(counts["Neutral"]/total*100, 1), round(counts["Negative"]/total*100, 1)

    base_p, base_n, base_neg = calc_pct(baseline_counts)
    prop_p, prop_n, prop_neg = calc_pct(proposed_counts)
    
    print("\n" + "="*50)
    print("🏆 實驗結果統計 (可直接寫入論文 3.2 節)")
    print("="*50)
    print(f"總測試天數: {TEST_DAYS} 天")
    print(f"原始蒐集總筆數: {total_raw} 筆 (每日平均 {daily_avg} 筆)")
    print(f"去重後保留筆數: {total_dedup} 筆")
    print(f"LLM 過濾後筆數: {total_llm} 筆 (剔除了 {total_dedup - total_llm} 筆雜訊)")
    print("-" * 50)
    print(f"【對照組】傳統原始 FinBERT 情緒分佈:")
    print(f"正向: {base_p}%, 中性: {base_n}%, 負向: {base_neg}%")
    print(f"【實驗組】本研究提出機制 (混合權重) 情緒分佈:")
    print(f"正向: {prop_p}%, 中性: {prop_n}%, 負向: {prop_neg}%")
    print("="*50)
    
    # 輸出 CSV
    df = pd.DataFrame(results)
    df.to_csv("ab_test_experiment_results.csv", index=False, encoding="utf-8-sig")
    print(f"💾 詳細對比報表已儲存至 'ab_test_experiment_results.csv'，共耗時 {round(time.time()-t_start, 1)} 秒。")

if __name__ == "__main__":
    run_experiment()