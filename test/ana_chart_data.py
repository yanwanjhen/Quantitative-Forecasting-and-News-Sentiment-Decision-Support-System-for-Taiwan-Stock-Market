import pandas as pd

df = pd.read_csv('experiment_results_2026.csv')
total = len(df)

base_pos = (df['LLM_filtered_FinBERT_Label'] == 'Positive').sum()
base_neu = (df['LLM_filtered_FinBERT_Label'] == 'Neutral').sum()
base_neg = (df['LLM_filtered_FinBERT_Label'] == 'Negative').sum()

prop_pos = (df['LLM_filtered_Hybrid_Label'] == 'Positive').sum()
prop_neu = (df['LLM_filtered_Hybrid_Label'] == 'Neutral').sum()
prop_neg = (df['LLM_filtered_Hybrid_Label'] == 'Negative').sum()

print("Base:", base_pos, base_neu, base_neg)
print("Prop:", prop_pos, prop_neu, prop_neg)
