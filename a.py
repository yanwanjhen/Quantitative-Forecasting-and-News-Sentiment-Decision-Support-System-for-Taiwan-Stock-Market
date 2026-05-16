from graphviz import Digraph

dot = Digraph("system_architecture_formal", format="svg")

# ===== 全域設定 =====
dot.attr(
    rankdir="LR",
    bgcolor="white",
    splines="ortho",
    nodesep="0.55",
    ranksep="0.85",
    fontname="Microsoft JhengHei",
    compound="true"
)

# ===== 節點樣式 =====
dot.attr(
    "node",
    shape="rect",
    style="rounded,filled",
    fontname="Microsoft JhengHei",
    fontsize="12",
    color="#2F4F6F",
    penwidth="1.3",
    fillcolor="#FFFFFF",
    margin="0.18,0.12"
)

# ===== 線條樣式 =====
dot.attr(
    "edge",
    color="#2F4F6F",
    penwidth="1.2",
    arrowsize="0.75",
    fontname="Microsoft JhengHei",
    fontsize="11"
)

# ===== 主流程節點 =====
dot.node(
    "input",
    "使用者輸入\n\n股票代碼／公司名稱\n台股大盤／金融術語",
    fillcolor="#F8F9FA"
)

dot.node(
    "frontend",
    "React 前端互動介面\n\n問題輸入\n風險設定\n分析結果呈現",
    fillcolor="#F8F9FA"
)

dot.node(
    "intent",
    "意圖辨識與標的解析\n\n判斷問題類型\n解析分析標的",
    fillcolor="#F8F9FA"
)

# ===== 股價分析模組 =====
with dot.subgraph(name="cluster_price") as c:
    c.attr(
        label="股價分析模組",
        fontname="Microsoft JhengHei",
        fontsize="14",
        color="#2F4F6F",
        penwidth="1.4",
        style="rounded",
        bgcolor="#FFFFFF"
    )

    c.node(
        "price_data",
        "歷史價量資料",
        fillcolor="#EEF3F7"
    )

    c.node(
        "price_process",
        "技術指標計算\n+\nLSTM 量化預測",
        fillcolor="#EEF3F7"
    )

    c.node(
        "price_output",
        "股價分析結果\n\n量化訊號\n技術狀態\n風險指標",
        fillcolor="#EEF3F7"
    )

    c.edge("price_data", "price_process")
    c.edge("price_process", "price_output")

# ===== 新聞情緒模組 =====
with dot.subgraph(name="cluster_news") as c:
    c.attr(
        label="新聞情緒模組",
        fontname="Microsoft JhengHei",
        fontsize="14",
        color="#2F4F6F",
        penwidth="1.4",
        style="rounded",
        bgcolor="#FFFFFF"
    )

    c.node(
        "news_data",
        "近期新聞資料",
        fillcolor="#EEF3F7"
    )

    c.node(
        "news_process",
        "LLM 實體過濾\n+\nHybrid FinBERT 分析",
        fillcolor="#EEF3F7"
    )

    c.node(
        "news_output",
        "新聞情緒結果\n\n情緒分數\n情緒標籤\n新聞關聯性",
        fillcolor="#EEF3F7"
    )

    c.edge("news_data", "news_process")
    c.edge("news_process", "news_output")

# ===== 整合與輸出 =====
dot.node(
    "llm",
    "LLM 自然語言說明模組\n\n整合股價分析、新聞情緒\n與使用者風險偏好",
    fillcolor="#F8F9FA"
)

dot.node(
    "result",
    "系統回覆與結果呈現\n\n自然語言分析\n模型訊號\n新聞情緒解釋",
    fillcolor="#F8F9FA"
)

# ===== 主流程連線 =====
dot.edge("input", "frontend")
dot.edge("frontend", "intent")

dot.edge("intent", "price_data")
dot.edge("intent", "news_data")

dot.edge("price_output", "llm")
dot.edge("news_output", "llm")

dot.edge("llm", "result")

# ===== 輸出 =====
dot.render("system_architecture_formal", cleanup=True)

dot.format = "png"
dot.render("system_architecture_formal", cleanup=True)

print("已輸出：system_architecture_formal.svg 與 system_architecture_formal.png")