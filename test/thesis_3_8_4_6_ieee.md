# 3.8 系統實作與介面設計 (System Implementation and Interface Design)

本節將詳細說明如何將不同分析模組整合至一個統一的互動式資訊系統中。該系統的主要目標是整合多源異質數據，並透過直觀的使用者介面提供深度市場分析，從而幫助多樣化背景之投資人進行決策輔助。

## 3.8.1 系統整體架構

本研究所提之投資輔助系統，採取高度模組化且低耦合的架構設計。為實現從數據擷取至自然語言報告生成之端到端 (End-to-End) 流程，系統依序整合下列七項核心模組：

1. **LSTM 量化預測模組 (LSTM Quantitative Prediction Module)**：基於長短期記憶網路，負責接收並分析歷史價量資料 (至少涵蓋 60 日輸入特徵)。輸出包含：預期報酬率 (Predicted Return)、建議交易訊號 (Signal，如買入、賣出或觀望)、市場狀態 (Regime，如多頭強勢、震盪等)，以及近一年最大回撤 (Max Drawdown) 等核心風險指標。
2. **技術指標分析 (Technical Indicator Analysis)**：針對給定的標的，預先計算包含對數報酬率 (Log Return)、波動率 (Volatility)、相對強弱指標 (RSI)、布林通道寬度 (BB Width) 及趨勢指標 (ADX) 等量化特徵，經過標準化與離群值處理後，輸入至深度學習模型。
3. **新聞情緒分析模組 (News Sentiment Analysis Module)**：自動抓取 Google News 等主要財經新聞來源，為評估市場外部消息面提供非結構化文字數據。
4. **LLM 實體過濾 (LLM Entity Filtering)**：為解決傳統情緒分析的雜訊問題，本系統引入 LLM (如 Gemini/GPT) 針對新聞標題與內文進行具名實體識別 (Named Entity Recognition, NER)。設定嚴格的過濾規則，剔除包含「房市」、「套房」、「建案」等無關新聞，並精準鎖定使用者查詢的目標公司或大盤，確保進入後續模型的文本高純度。
5. **Hybrid FinBERT 情緒分數 (Hybrid FinBERT Sentiment Scoring)**：結合金融領域微調模型 FinBERT 與領域字典關鍵字 (投資、營收、EPS等)。透過計算正負向詞彙命中數，並使用雙曲正切函數 (Hyperbolic Tangent, tanh) 將情緒分數平滑標準化至 [-1, 1] 區間，負數偏空，正數偏多。
6. **LLM 自然語言說明 (LLM Natural Language Explanation)**：在此階段，LLM 扮演了解釋層 (Explanation Layer) 的角色，而非單純的預測代理。它會接收包含：技術狀態、預期報酬、情緒分數 (Sentiment Score) 及其它原始新聞文字，根據投資人的風險偏好 (如：穩健、保守、積極)，轉譯為具語義邏輯且量身打造的綜合分析報告。
7. **前端介面展示 (Frontend Interface)**：採用互動式網頁架構 (React/Streamlit 架構理念) 構建前端 Web 介面。透過對話框輸入與動態更新的視覺化儀表板 (Dashboard) 並行，整合所有分析圖表與數值結果輸出。

*(Fig. 1 系統整體架構圖請置於此處)*
> **Fig. 1.** Overall system architecture integrating quantitative prediction, sentiment analysis, and LLM-based processing modules.

## 3.8.2 前端介面設計與功能區塊

為降低模型黑箱問題 (Black-box Problem) 並提升資訊吸收效率，本研究之介面設計導入了模組化的功能區塊。傳統系統多單純顯示指標庫，本系統則將資料進行高階封裝，其核心功能區塊與設計職責詳見 Table I。

> **Table I.** Functional blocks and descriptions of the user interface.

| 介面區塊 (Interface Block) | 功能說明與實作細節 (Functional Description) |
| :--- | :--- |
| **股票查詢區 (Stock Query Area)** | 透過輸入框取得自然語言查詢。後端實作 `extract_intent`，透過正則表達式與 LLM 解析出台灣股票代碼 (e.g., 2330.TW)、市場大盤或金融名詞的解釋意圖。 |
| **量化結果區 (Quantitative Results Area)** | 以「儀表板 (Metric Dashboard)」顯示：①預期報酬、②綜合訊號 (買入/賣出/觀望)、③市場狀態 (趨勢)、④歷史近一年最大回撤 (Max DD)。並具備 K 線與走勢圖表，依 1M, 3M, 1Y 區間動態呈現價格動能。 |
| **新聞情緒區 (News Sentiment Area)** | 顯示近五日過濾後之有效「新聞數量」，以及範圍於 [-1, 1] 之「情緒分數」。並於介面下方條列近期重大新聞列表及其各自被評估之極性與關聯詞。 |
| **LLM 說明區 (LLM Explanation Area)** | 透過漸進式串流 (Streaming) 文字生成技術，即時呈現 LLM 的口語化報告，剖析當代標的的財報層面、技術層面及短線操作建議策略。 |
| **動態追問區 (Dynamic Follow-ups)** | 實作 `buildFollowUps` 演算法，依據單股或多檔股票情境 (如比較兩檔股票)，自動生成關聯度高的下一階段提問，延續對話互動。 |
| **風險提示區 (Risk Prompt Area)** | 實作 `evaluateConsistency` 及 `evaluateTrust` 演算法。比較「量化訊號方向」與「情緒分數正負」，若出現嚴重分歧 (Divergence)，以醒目橫幅警告使用者短線不確定性極高。 |

## 3.8.3 自然語言分析與風險提示 (一致性驗證機制)

本研究系統最核心的設計理念之一在於**不直接提供絕對的買賣建議 (Non-deterministic trading recommendation)**，而是將複雜的量化參數、圖表技術狀態以及經過評分的語義情緒，收斂組合為一份「具備風險意識的自然語言分析」。

在風險提示機制上，系統引入了「信任與一致性校準 (Trust & Consistency Calibration)」演算法 (例如實作中的 `evaluateTrust` 與 `evaluateConsistency` 模組)。系統不預設任何單一模型為絕對真理：
- **一致性情境 (Aligned)**：水量化訊號為偏多 (如：買入、強烈買入、反彈契機)，且情緒分數亦為顯著正向 (如：大於 0.15) 時，系統將標記為「一致」，LLM 也將於報告中表示訊號互相支持。
- **分歧情境 (Conflicted)**：當發生量化訊號看多，但新聞情緒過冷 (分數 < -0.05)；或量化訊號看空，但新聞極度狂熱的情境時，系統防呆機制將被觸發。此時，「建議可信度」會自動降級至「中」或「低」。
- **下行風險防護 (High Drawdown)**：系統若偵測到近一年的最大回撤 (Max Drawdown) 劣於 -20%，即使訊號同向，也會強制發出「波動風險條件偏弱，建議先用風險控管做保護」的警語。

透過以此演算法驅動中介的風險提示，能有效協助使用者理解目前市場多空訊號的防禦共識，避免盲從所謂的 AI 神諭 (AI Oracle) 或單一指標。

---

# 4.6 系統展示與應用情境分析 (System Demonstration and Application Scenario Analysis)

為驗證本投資輔助系統在處理實際市場數據中的應用價值，本節展示該系統於真實情境下之跨模組運作流程與實際介面截圖回饋。

## 4.6.1 使用流程與查詢案例展示

系統的標準使用流程始於使用者在查詢區端輸入標的資訊。以查詢特定個股 (例如要求系統分析：台積電) 或台股大盤加權指數為案例探討：
1. 使用者在前端輸入：「請幫我分析近期台積電目前的量化訊號與走勢如何？」。
2. 系統後端之 `extract_intent` 攔截到 "台積電" 與相關代碼 "2330.TW"。
3. 系統即時啟動非同步作業，分別透過 Yahoo Finance API 擷取歷史股價，並向新聞 RSS 索取近五日內台積電相關報導。此時前端介面會顯示階段式進度指標 (如「抓取歷史價量中...」、「生成分析報告中...」)，使分析流程維持高透明度。

*(Fig. 2 介面截圖：使用者輸入與系統動態回應狀態請置於此處)*
> **Fig. 2.** Screenshot of the progressive system response interface upon query initiation, displaying real-time data fetching steps.

## 4.6.2 多模組分析結果呈現 (Dashboard Presentation)

在量化運算與新聞正規化處理完成後，系統畫面將同步呈現以下兩個面向之視覺化結果：

1. **量化的技術與風險指標**：介面利用 Dashboard Cards 動態渲染出數值指標。例如，預判該股「綜合訊號：買入」、「市場狀態：多頭強勢」、「預期報酬：1.25%」以及「歷史最大回撤：-15.0%」。並在下方呈現最近 3 個月 (3M) 的收盤價與 K 線圖，供投資者直接對照。
2. **純淨化的新聞情緒指標**：計算出本期台積電之綜合「情緒分數」(如：0.35)。此外，儀表板下方將依序列出經過 LLM 過濾後的原萃新聞標題與網址，標示各條新聞的正面/負面詞彙命中結果，以落實資訊可溯源性 (Traceability)。

*(Fig. 3 量化儀表板與新聞情緒條列之介面截圖請置於此處)*
> **Fig. 3.** Dashboard output integrating quantitative signal metrics, maximum drawdown, chart components, and detailed news sentiment breakdowns.

## 4.6.3 多檔標的比較與互動式追問情境 (Multi-target Comparison and Interactive Follow-ups)

除了單一個股與大盤分析外，本系統亦支援**多檔股票之橫向比較 (Portfolio Analysis)**。
1. **多標的意圖解析**：當使用者輸入如「請幫我比較長榮與陽明的差異」時，系統之 `extract_intent` 中的正則式與分類器會同時捕捉多個具名實體 (Entities)。
2. **平行運算與比較渲染**：系統將平行呼叫量化與情緒模組，將多檔標的之「預測報酬率」、「市場狀態」及「情緒分數」彙總後，交由專責的 `generate_portfolio_analysis_stream` 模組生成比較型自然語言分析。
3. **動態追問推論 (Dynamic Follow-ups)**：為延續對話上下文，系統實作了 `buildFollowUps` 演算法。當偵測到多標的比較時，介面底部會自動生成引導式追問，例如「長榮和陽明如果只能選一檔，現階段我該優先哪一檔？」或「這幾檔當中哪一檔最不適合追高？」。此設計大幅降低了使用者的語法構造負擔，達成連貫且深入的決策輔助。

*(Fig. 4 多檔標的比較與動態追問介面截圖請置於此處)*
> **Fig. 4.** Portfolio analysis layout and dynamically generated follow-up interactions for multi-stock comparisons.

## 4.6.4 LLM 深度分析生成與風險矛盾實戰

在此應用情境的最末階段，系統並不會任由 LLM 即興發揮，而是將上述定量的結果作為強限制條件 (Hard Constraints) 注入給 LLM (Prompt injection)。LLM 的報告產出將依循嚴厲的三步驟：1. 破題與情緒定調、2. 深度影響評估、3. 行動與應對建議。

**實際案例：發生訊號分歧時的風險提示展示**
為證明系統處理雜訊的能力，設定某一情境中，由於技術面漲多拉回，量化模組之綜合訊號給出「賣出 (Sell)」建議；然而新聞媒體由於落後效應，其情緒分數依然高達 `0.45 (強烈樂觀)`。
當此種矛盾發生時，介面渲染邏輯 (evaluateConsistency) 將判斷為 Conflicted。介面會升起警告 Banner，標示：`一致性檢查：訊號分歧 — 量化訊號與市場情緒並未同向，短線決策不確定性較高`。
同時，LLM 的自然語言說明段落將主動融合此一警告指出：「儘管媒體報端充斥樂觀營收預期，推升了高額的情緒分數，但量化與技術圖表已進入超買區並顯現修正的賣出訊號。兩端數據產生了顯著的背離 (Divergence)。作為穩健型投資人，系統強烈建議目前不宜過度追高，警惕新聞面的延遲與誘多效應。」

此設計不僅為冷冰冰的數據提供了具有人性的邏輯解釋，也確保系統達成輔助決策的最佳實用邊界。

*(Fig. 5 風險分歧橫幅提示與 LLM 自然語言警示畫面請置於此處)*
> **Fig. 5.** System layout demonstrating the generation of risk alerts and LLM explanatory warnings when experiencing a divergence between quantitative sell signals and overly optimistic news sentiment.
