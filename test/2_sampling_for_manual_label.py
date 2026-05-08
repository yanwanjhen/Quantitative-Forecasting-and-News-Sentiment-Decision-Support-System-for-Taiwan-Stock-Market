import pandas as pd

# 讀取剛剛跑完的總表
df = pd.read_csv("experiment_results_2026.csv")

# 解決 Gap 1：抽取 300 筆 (若總數不足 300 則全抽)
sample_size = min(300, len(df))
sample_df = df.sample(n=sample_size, random_state=42).copy()

# 新增讓您人工標註的欄位
sample_df['Manual_Label'] = '' 

# 匯出給您填寫
sample_df.to_csv("manual_label_sample_300.csv", index=False, encoding="utf-8-sig")

print(f"🎯 已成功抽出 {sample_size} 筆新聞至 'manual_label_sample_300.csv'。")
print("👉 請打開它，在 'Manual_Label' 填入 Positive, Neutral 或 Negative。")
print("👉 填寫完成後，請另存為 'manual_label_sample_300_filled.csv'")