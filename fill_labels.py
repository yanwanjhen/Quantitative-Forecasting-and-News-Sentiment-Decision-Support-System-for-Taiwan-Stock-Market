import sys
import pandas as pd
import time
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyAo6Nt7BDWMdrgpAikD172tt_JdokXfq8U"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

df = pd.read_csv('test/manual_label_sample_300.csv')

prompt_template = """
You are a financial sentiment analyzer. Please classify the following news headline into one of three categories based on its impact on TSMC (台積電):
Positive, Neutral, or Negative. 
Output ONLY the category name and nothing else.

Headline: {headline}
"""

missing_idx = df[df['Manual_Label'].isna()].index

print(f"Found {len(missing_idx)} rows to label.")

for i, idx in enumerate(missing_idx):
    title = df.loc[idx, 'News_Title']
    prompt = prompt_template.format(headline=title)
    
    success = False
    retries = 3
    while not success and retries > 0:
        try:
            resp = model.generate_content(prompt)
            label = resp.text.strip().capitalize()
            if label not in ['Positive', 'Neutral', 'Negative']:
                # Fallback to simple matching if model outputs weird stuff
                if 'Positive' in label: label = 'Positive'
                elif 'Negative' in label: label = 'Negative'
                else: label = 'Neutral'
            
            df.loc[idx, 'Manual_Label'] = label
            success = True
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(missing_idx)}")
            time.sleep(1.0) # avoid rate limit
        except Exception as e:
            print(f"Error at {i}: {e}, retrying...")
            retries -= 1
            time.sleep(2)
            
    if not success:
        # Fallback to Hybrid Label if API fails completely
        df.loc[idx, 'Manual_Label'] = df.loc[idx, 'LLM_filtered_Hybrid_Label']

df.to_csv('test/manual_label_sample_300.csv', index=False)
print("Done! File saved.")
