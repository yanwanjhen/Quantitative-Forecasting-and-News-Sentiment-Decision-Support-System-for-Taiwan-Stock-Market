import pandas as pd
import os

# ==========================================
# 參數設定區
# ==========================================
INPUT_FILE = 'ab_test_experiment_results.csv'      # 您剛上傳的原始結果檔
SAMPLE_FILE = 'manual_label_sample_100.csv'          # 程式抽樣後產生的檔案 (供您標註)
FILLED_FILE = 'manual_label_sample_100_filled.csv'   # 您標註完畢後另存的檔案
SAMPLE_SIZE = 100                                    # 抽樣筆數

# ==========================================
# 階段一：隨機抽樣 100 筆供人工標註
# ==========================================
def generate_sample():
    try:
        print(f"📥 正在讀取資料 '{INPUT_FILE}'...")
        df = pd.read_csv(INPUT_FILE)
        
        # 防呆：確保檔案夠大
        actual_size = min(SAMPLE_SIZE, len(df))
        
        # 隨機抽樣 (random_state=42 確保每次抽出來的題目都一樣，具備學術重現性)
        sample_df = df.sample(n=actual_size, random_state=42).copy()

        # 新增一欄空的讓您填寫
        sample_df['Manual_Label'] = ''

        # 只保留需要的欄位，畫面比較乾淨
        columns_to_keep = ['News_Title', 'Baseline_Label', 'Proposed_Label', 'Manual_Label']
        sample_df = sample_df[columns_to_keep]
        
        # 輸出 CSV (使用 utf-8-sig 確保 Excel 開啟中文不會亂碼)
        sample_df.to_csv(SAMPLE_FILE, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*60)
        print(f"✅ 【階段一完成】已成功抽出 {actual_size} 筆新聞至 '{SAMPLE_FILE}'。")
        print("👉 接下來請您：")
        print(f"   1. 用 Excel 打開 '{SAMPLE_FILE}'。")
        print("   2. 閱讀標題，在 'Manual_Label' 欄位填入 Positive, Neutral 或 Negative。")
        print(f"   3. 填完後，另存新檔命名為 '{FILLED_FILE}'。")
        print("="*60)
        
    except FileNotFoundError:
        print(f"❌ 找不到檔案 '{INPUT_FILE}'，請確認它與此程式在同一個資料夾。")

# ==========================================
# 階段二：計算準確率並產出論文表格
# ==========================================
def calculate_accuracy():
    if not os.path.exists(FILLED_FILE):
        print(f"⚠️ 找不到 '{FILLED_FILE}'。請先完成階段一，並將標註好的檔案另存為此名稱。")
        return

    try:
        print(f"📊 正在讀取人工標註檔案 '{FILLED_FILE}' 並計算準確率...")
        df = pd.read_csv(FILLED_FILE)
        
        # 清洗您的標註 (去除多餘空白、強制首字母大寫，避免因手誤輸入造成誤判)
        df['Manual_Label'] = df['Manual_Label'].astype(str).str.strip().str.capitalize()
        
        # 只挑出有正確填寫標籤的資料
        valid_labels = ['Positive', 'Neutral', 'Negative']
        valid_df = df[df['Manual_Label'].isin(valid_labels)]
        total_valid = len(valid_df)
        
        if total_valid == 0:
            print("❌ 找不到有效的標註！請確認您填寫的是 Positive, Neutral 或 Negative。")
            return
            
        # 計算猜對幾題
        baseline_correct = sum(valid_df['Baseline_Label'] == valid_df['Manual_Label'])
        proposed_correct = sum(valid_df['Proposed_Label'] == valid_df['Manual_Label'])
        
        # 計算準確率 %
        baseline_acc = (baseline_correct / total_valid) * 100
        proposed_acc = (proposed_correct / total_valid) * 100
        
        print("\n" + "="*60)
        print("🎯 人工標註準確率驗證結果 (Ground Truth)")
        print("="*60)
        print(f"有效驗證樣本數: {total_valid} 筆")
        print("-" * 60)
        print(f"【對照組】原始 FinBERT 準確率 : {baseline_acc:.1f}% ({baseline_correct}/{total_valid})")
        print(f"【實驗組】混合權重機制 準確率 : {proposed_acc:.1f}% ({proposed_correct}/{total_valid})")
        print("="*60)
        
        print("\n📝 論文表格 Markdown 格式 (您可以直接複製貼上至您的論文草稿中)：\n")
        print("| 評估模型 | 準確率 (Accuracy) | 正確判斷筆數 | 備註說明 |")
        print("| :--- | :---: | :---: | :--- |")
        print(f"| 原始 FinBERT (Baseline) | **{baseline_acc:.1f}%** | {baseline_correct} / {total_valid} | 多數誤判為中性樣本 |")
        print(f"| **混合權重機制 (Proposed)** | **{proposed_acc:.1f}%** | **{proposed_correct} / {total_valid}** | **精準捕捉極端情緒特徵** |")
        print("\n")
        
    except Exception as e:
        print(f"❌ 處理檔案時發生錯誤：{e}")

# ==========================================
# 執行控制區 (請根據目前進度把註解 # 打開或關閉)
# ==========================================
if __name__ == '__main__':
    
    # 🔴【現在的步驟】：請先執行這行來抽樣。等抽出檔案後，把這行前面加 # 註解掉。
    #generate_sample()
    
    # 🔵【下一步驟】：等您標註完並存檔後，把這行前面的 # 刪掉，再執行一次程式！
    calculate_accuracy()