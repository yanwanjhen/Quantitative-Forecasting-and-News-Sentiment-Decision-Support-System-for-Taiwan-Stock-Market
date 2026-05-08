import urllib.parse
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time

keyword = "台積電"
headers = {'User-Agent': 'Mozilla/5.0'}
all_titles = []

start_date = datetime(2026, 2, 1)
end_date = datetime(2026, 4, 30)
current_date = start_date

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
        time.sleep(0.5)
    except Exception as e:
        pass
    current_date = next_date

print("1. 原始抓取:", len(all_titles))
print("去重後 (僅set):", len(list(set(all_titles))))
