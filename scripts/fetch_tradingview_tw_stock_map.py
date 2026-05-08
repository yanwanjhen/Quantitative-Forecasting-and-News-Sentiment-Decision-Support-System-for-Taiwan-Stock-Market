import json
from pathlib import Path

import requests
import csv
import io


def build_ticker(exchange: str, code: str) -> str:
    exchange = (exchange or "").upper()
    code = str(code).strip()
    if exchange in {"TWSE", "TSE"}:
        return f"{code}.TW"
    if exchange in {"TPEX", "OTC"}:
        return f"{code}.TWO"
    # Fallback: most TW market-mover pages use TWSE symbols
    return f"{code}.TW"

def fetch_twse_listed_names() -> dict[str, str]:
    """
    Returns mapping: code -> Chinese short name (公司簡稱).
    Source: TWSE OpenAPI (JSON).
    """
    url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    data = resp.json()
    out: dict[str, str] = {}
    for row in data:
        code = str(row.get("公司代號", "")).strip()
        short = str(row.get("公司簡稱", "")).strip()
        if code and short:
            out[code] = short
    return out


def fetch_tpex_otc_names() -> dict[str, str]:
    """
    Returns mapping: code -> Chinese short name (公司簡稱).
    Source: TPEx basic data CSV (BIG5).
    """
    url = "http://dts.twse.com.tw/opendata/t187ap03_O.csv"
    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    text = resp.content.decode("big5", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    out: dict[str, str] = {}
    for row in reader:
        code = str(row.get("公司代號", "")).strip()
        short = str(row.get("公司簡稱", "")).strip()
        if code and short:
            out[code] = short
    return out


def main() -> None:
    url = "https://scanner.tradingview.com/taiwan/scan"
    headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}

    # Columns order matches earlier experiments:
    # [name, description, logoid, type, exchange, sector, industry]
    columns = ["name", "description", "logoid", "type", "exchange", "sector", "industry"]

    # First request gives us totalCount.
    first_body = {
        "filter": [],
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": columns,
        "sort": {"sortBy": "name", "sortOrder": "asc"},
        "range": [0, 1],
    }
    first = requests.post(url, headers=headers, data=json.dumps(first_body), timeout=30)
    first.raise_for_status()
    payload = first.json()
    total = int(payload.get("totalCount", 0))
    if total <= 0:
        raise RuntimeError(f"Unexpected totalCount={total}")

    out: dict[str, list[str]] = {}
    twse_names = fetch_twse_listed_names()
    tpex_names = fetch_tpex_otc_names()
    page_size = 500
    for start in range(0, total, page_size):
        body = {
            "filter": [],
            "symbols": {"query": {"types": []}, "tickers": []},
            "columns": columns,
            "sort": {"sortBy": "name", "sortOrder": "asc"},
            "range": [start, min(start + page_size - 1, total - 1)],
        }
        resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=60)
        resp.raise_for_status()
        chunk = resp.json().get("data", [])
        for row in chunk:
            d = row.get("d") or []
            if len(d) < 6:
                continue
            code = str(d[0]).strip()
            desc = str(d[1]).strip()
            exchange = str(d[4]).strip()
            if not code:
                continue
            ticker = build_ticker(exchange, code)
            zh = twse_names.get(code) or tpex_names.get(code) or ""
            canonical = zh or desc or code

            # Keys: English description (from TradingView) and code itself.
            # Values follow `data_fetch.py` external map format: [ticker, canonical_name]
            if desc:
                out.setdefault(desc, [ticker, canonical])
            out.setdefault(code, [ticker, canonical])
            if zh:
                out.setdefault(zh, [ticker, zh])

    target = Path(__file__).resolve().parents[1] / "data" / "tw_stock_map.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} mappings to {target}")


if __name__ == "__main__":
    main()
