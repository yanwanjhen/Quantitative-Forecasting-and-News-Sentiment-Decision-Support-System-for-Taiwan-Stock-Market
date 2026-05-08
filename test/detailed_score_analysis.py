import torch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. 載入模型
# ==========================================
print("🧠 正在載入 yiyanghkust/finbert-tone-chinese 模型...")
model_name = "yiyanghkust/finbert-tone-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
finbert_model.to(device)
finbert_model.eval()

# ==========================================
# 2. 定義情緒詞典 
# ==========================================
positive_words = [
    "大漲", "上揚", "走揚", "創高", "反彈", "翻倍", "新高", "收紅", "填息", "帶量", 
    "噴發", "站穩", "續強", "轉強", "漲停", "站回", "收復", "再漲", "衝上", "漲逾", 
    "強彈", "衝高", "穩守", "撐腰", "紅燈", "亮紅燈", "漲", "領航", "強勢", "有戲", 
    "強升", "登高", "點火", "回溫", "暴衝", "飆", "強攻", "看俏", "創新高", "飆漲",
    "成長", "利多", "看好", "突破", "獲利", "增長", "上修", "升溫", "強勁", "優於", 
    "增加", "提升", "回升", "轉盈", "亮眼", "達標", "滿載", "營收", "創紀錄", "受惠", 
    "攜手", "結盟", "年增", "月增", "擴增", "強化", "私募", "入股", "大咖", "綁樁", 
    "震撼彈", "缺貨", "缺爆", "搶單", "奪下", "通吃", "大單", "急單", "護盤", "雙成長", 
    "添柴", "燒", "佳", "旺", "優", "報喜", "繳出", "每股賺", "賺", "樂觀", "積極", 
    "正面", "買超", "加碼", "敲進", "回補", "進駐", "連買", "買單", "掃貨", "大買", 
    "回流", "注資", "抄底", "力挺", "狂湧", "青睞", "照顧"
]

negative_words = [
    "下跌", "大跌", "重挫", "暴跌", "下滑", "腰斬", "新低", "走低", "探底", "貼息", 
    "收黑", "摜破", "失守", "承壓", "遇阻", "跌穿", "跌停", "回檔", "修正", "疲軟",
    "震盪", "翻黑", "得而復失", "熄火", "平盤", "跌", "綠燈", "亮綠燈", "崩", "殺",
    "吞跌停", "下探", "未站穩", "崩跌", "摔", "狂震", "慘墜", "陣亡", "探低", "破底", 
    "重摔", "利空", "衰退", "看淡", "虧損", "下修", "降溫", "疲弱", "不如", "減少", 
    "下降", "轉虧", "慘淡", "年減", "月減", "不如預期", "蒸發", "縮水", "砍單", "瓶頸", 
    "示警", "衝擊", "危機", "重創", "遭殃", "掉單", "流失", "保守", "裁員", "違約", 
    "調查", "訴訟", "停滯", "失利", "堪憂", "降評", "挨刀", "每股虧", "虧", "差", 
    "弱", "損", "悲觀", "消極", "負面", "賣壓", "賣超", "減持", "出脫", "拋售", "撤資", 
    "連賣", "倒貨", "提款", "瘋狂拋售", "棄守", "調節", "狂砍"
]

ambiguous_positive_words = {"攜手", "突破", "優", "賺", "飆", "大咖", "震撼彈", "照顧", "佳", "旺", "燒"}
finance_context_terms = {"營收", "獲利", "財報", "法說", "訂單", "供應鏈", "股價", "個股", "股市", "股票", "股東", "台股", "EPS", "殖利率", "本益比", "投資", "配息", "除息", "買超", "賣超", "市值", "大盤", "外資", "法人"}
non_investment_context_terms = {"基金會", "文教", "職缺", "員工", "市長", "城市", "浪漫", "健康", "候選", "永續", "公益", "校園", "徵才", "招募", "表白", "董事", "活動", "供應商"}
negative_keyword_exclusions = {"下調": ("下調漲",)}


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

# ==========================================
# 3. 核心計分函數 (回傳字典供 DataFrame 使用)
# ==========================================
def calculate_both_scores(text, target_company=None):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return None

    clean_title = re.sub(r'^.*?》|《.*?》|【.*?】|〈.*?〉|「.*?」|『.*?』|快訊／|- 上市櫃', '', text).strip()
    
    if target_company and target_company not in ["大盤", "台股", "加權指數"]:
        segments = re.split(r'[，,。：:；;！!？?、\s]+', clean_title)
        relevant_segments = [seg for seg in segments if target_company in seg]
        target_text = " ".join(relevant_segments) if relevant_segments else clean_title
    else:
        target_text = clean_title

    inputs = tokenizer(target_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    logits = outputs.logits[0]
    
    l_neu, l_pos, l_neg = logits[0].item(), logits[1].item(), logits[2].item()
    p_neu = torch.nn.functional.softmax(logits, dim=-1)[0].item()
    
    # ----------------------------------------------------
    # [機制 1] 傳統 FinBERT 原始分數 (Baseline - 使用最原始機率法)
    # ----------------------------------------------------
    probs = torch.nn.functional.softmax(logits, dim=-1)
    class_idx = torch.argmax(probs).item()
    
    # 決定 Baseline 標籤 (純看機率最高者)
    if class_idx == 1: 
        baseline_label = "Positive"
    elif class_idx == 2: 
        baseline_label = "Negative"
    else: 
        baseline_label = "Neutral"
        
    # 為了在 CSV 中仍能看到分數，這裡保留原始的相對強度計算，但不影響標籤判定
    raw_score = l_pos - l_neg
    baseline_score = torch.tanh(torch.tensor(raw_score / 1.5)).item()
    baseline_score = round(baseline_score, 4)

    # ----------------------------------------------------
    # [機制 2] 混合權重 (Proposed)
    # ----------------------------------------------------
    pos_hits, neg_hits = _keyword_hits(target_text)
    keyword_score = torch.tanh(torch.tensor((len(pos_hits) - len(neg_hits) * 1.2) / 2.0)).item() 

    if p_neu > 0.5:
        w_model, w_keyword = 0.3, 0.7
    else:
        w_model, w_keyword = 0.8, 0.2
        
    proposed_score = (baseline_score * w_model) + (keyword_score * w_keyword)
    
    if abs(keyword_score) > 0.7 and abs(proposed_score) < 0.3:
         proposed_score = (proposed_score + keyword_score) / 2

    if _is_non_investment_context(target_text) and not _is_finance_context(target_text):
        proposed_score *= 0.5
            
    proposed_score = max(min(proposed_score, 1.0), -1.0)
    proposed_score = round(proposed_score, 4)
    
    # 採用您修改後的 ±0.15 健康門檻
    if proposed_score > 0.15: proposed_label = "Positive"
    elif proposed_score < -0.15: proposed_label = "Negative"
    else: proposed_label = "Neutral"
    
    return {
        "News_Title": text,
        "Cleaned_Text_Analyzed": target_text,
        "Baseline_Score": baseline_score,
        "Baseline_Label": baseline_label,
        "Proposed_Score": proposed_score,
        "Proposed_Label": proposed_label,
        "FinBERT_Neutral_Prob": round(p_neu, 3),
        "Positive_Keywords_Hit": "、".join(pos_hits) if pos_hits else "無",
        "Negative_Keywords_Hit": "、".join(neg_hits) if neg_hits else "無"
    }

# ==========================================
# 4. 讀取輸入 CSV，全部運算並輸出新 CSV
# ==========================================
if __name__ == "__main__":
    # 您可以根據要測個股或大盤來更改檔案名稱
    input_file = 'ab_test_experiment_results.csv' 
    output_file = 'detailed_score_comparison_full.csv'
    
    # ⚠️ 請記得更改這裡：如果測大盤請改為 "大盤"，測個股請填入公司名
    target = "台積電" 

    try:
        print(f"📥 正在讀取檔案: {input_file}")
        df = pd.read_csv(input_file)
        
        all_results = []
        total_rows = len(df)
        
        print(f"⏳ 開始處理 {total_rows} 筆新聞資料...")
        
        for idx, row in df.iterrows():
            news_title = row['News_Title']
            result = calculate_both_scores(news_title, target_company=target)
            
            if result:
                all_results.append(result)
                
            # 簡單進度條提示，避免畫面卡住
            if (idx + 1) % 50 == 0:
                print(f"   已處理 {idx + 1} / {total_rows} 筆...")

        # 將結果轉換為 DataFrame 並匯出
        output_df = pd.DataFrame(all_results)
        
        # 加上 utf-8-sig 確保 Excel 打開中文不會亂碼
        output_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        
        print("\n" + "="*60)
        print(f"✅ 處理完成！共成功分析並轉換 {len(output_df)} 筆資料。")
        print(f"💾 詳細對比結果已完整儲存至: {output_file}")
        print("="*60)
        
    except FileNotFoundError:
        print(f"⚠️ 找不到 '{input_file}'，請確認檔案名稱與路徑是否正確。")
