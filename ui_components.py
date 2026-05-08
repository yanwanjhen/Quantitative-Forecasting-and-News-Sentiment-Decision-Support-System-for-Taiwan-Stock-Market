import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import re

st.markdown("""
<style>
/* 針對 Metric 設定卡片視角 (Card UI) */
div[data-testid="stMetric"] {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 15px 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    border: 1px solid #f0f2f6;
    transition: transform 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.1);
}
/* 數值字體設定 */
div[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    color: #1f77b4; 
    font-weight: bold;
}
/* 標題小標籤設定 */
div[data-testid="stMetricLabel"] {
    font-size: 1rem;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

def plot_candlestick(df, ticker):
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], increasing_line_color='red', decreasing_line_color='green')])
    fig.update_layout(
        title=f'{ticker} 近一個月 K線圖', 
        xaxis_rangeslider_visible=False, 
        height=300, 
        margin=dict(l=10, r=10, t=40, b=10),
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    return fig

def plot_line_chart(df, ticker):
    fig = go.Figure(data=[go.Scatter(x=df.index, y=df['Close'], mode='lines', line=dict(color='#1f77b4', width=2))])
    fig.update_layout(
        title=f'{ticker} 近一個月收盤價走勢圖', 
        height=300, 
        margin=dict(l=10, r=10, t=40, b=10),
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    return fig

def _parse_percent(value):
    if value is None:
        return 0.0
    try:
        return float(re.sub(r"[()%]", "", str(value))) / 100
    except Exception:
        return 0.0

def evaluate_confidence(quant_data, news_data):
    signal = quant_data.get("signal", "觀望")
    sentiment_score = news_data.get("sentiment_score", 0.0)
    news_status = news_data.get("news_count_status", "sufficient")
    max_dd = _parse_percent(quant_data.get("max_dd", "0%"))

    bullish_signal = signal in ["買入", "強烈買入", "反彈契機"]
    bearish_signal = signal in ["賣出", "強烈賣出", "持倉 (超賣信號禁空)", "強烈觀望 (跌深抵達下軌)"]
    bullish_sentiment = sentiment_score > 0.15
    bearish_sentiment = sentiment_score < -0.05

    aligned = (bullish_signal and bullish_sentiment) or (bearish_signal and bearish_sentiment)
    conflicted = (bullish_signal and bearish_sentiment) or (bearish_signal and bullish_sentiment)
    weak_signal = signal in ["觀望", "資料過少", "無法預測", "集成失敗", "運算錯誤"]
    data_limited = news_status in ["none", "few"]
    high_drawdown = max_dd <= -0.2

    if aligned and news_status == "sufficient" and not high_drawdown:
        return "高", "量化與情緒同向，且新聞樣本足夠。", "normal"
    if conflicted or data_limited or high_drawdown:
        if conflicted:
            reason = "量化訊號與市場情緒分歧。"
        elif high_drawdown:
            reason = "近一年回撤偏大，波動風險較高。"
        else:
            reason = "新聞樣本不足，情緒代表性較弱。"
        return "低", reason, "inverse"
    if weak_signal:
        return "中", "訊號偏保守或資料條件普通。", "off"
    return "中", "量化或情緒僅單邊支持，建議搭配風險控管。", "off"

import numpy as np



def render_dashboard(db):
    st.session_state["_dashboard_render_counter"] = st.session_state.get("_dashboard_render_counter", 0) + 1
    chart_key_prefix = f"{db['ticker']}_{st.session_state['_dashboard_render_counter']}"

    st.markdown(f"### 🎯 {db['ticker']} 核心分析指標")
    investor_profile = db.get("investor_profile", {})
    if investor_profile:
        st.caption(
            f"投資人設定：{investor_profile.get('style', '穩健')}｜"
            f"最大可接受虧損 {investor_profile.get('max_loss_pct', 10)}%"
        )
    
    # 建立分頁
    tab1, tab2, tab3 = st.tabs(["📈 核心數據與線圖", "📰 情緒新聞清單", "🤖 量化模型參數"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        # 根據 signal 決定顏色：觀望用灰色，買入用紅色，賣出用綠色
        signal = db['quant_data']['signal']
        if signal == "觀望":
            delta_color_1 = "off"  # 灰色
        elif signal in ["買入", "強烈買入", "反彈契機"]:
            delta_color_1 = "normal"  # 紅色
        elif signal in ["賣出", "強烈賣出", "持倉 (超賣信號禁空)", "強烈觀望 (跌深抵達下軌)"]:
            delta_color_1 = "inverse"  # 綠色
        else:
            delta_color_1 = "off"  # 預設灰色
            
        col1.metric("預期報酬率", f"{db['quant_data']['predicted_return']*100:.3f}%", delta=signal, delta_color=delta_color_1)
        
        s_score = db['news_data']['sentiment_score']
        col2.metric("近期市場情緒", s_score, delta="偏多 (正向)" if s_score > 0 else "偏空 (負向)")    
        col3.metric("趨勢狀態", db['quant_data'].get('regime', '未知'), delta=f"近1年 Max DD: {db['quant_data'].get('max_dd', '0%')}", delta_color="inverse")
        confidence_level, confidence_reason, confidence_color = evaluate_confidence(db['quant_data'], db['news_data'])
        col4.metric("建議可信度", confidence_level, delta=confidence_reason, delta_color=confidence_color)

        quant_error = db['quant_data'].get("error_message")
        if quant_error:
            st.warning(f"量化模型狀態：{quant_error}")
        
        df_history = db['df_history']
        
        # --- 信任校準 (Trust Calibration) ---
        trust_warning = None
        if signal in ["買入", "強烈買入", "反彈契機"] and s_score < -0.2:
            trust_warning = "⚠️ **信心度警告**：技術量化訊號偏多，但近期市場情緒顯著偏空，建議保守評估。"
        elif signal in ["賣出", "強烈賣出", "持倉 (超賣信號禁空)", "強烈觀望 (跌深抵達下軌)"] and s_score > 0.2:
            trust_warning = "⚠️ **信心度警告**：技術量化訊號偏空或遇壓，但近期市場情緒火熱，需留意軋空風險或誘多陷阱。"
        
        if trust_warning:
            st.warning(trust_warning)
            
        if df_history is not None and not df_history.empty:
            fig_col1, fig_col2 = st.columns(2)
            with fig_col1:
                fig_k = plot_candlestick(df_history, db['ticker'])
                st.plotly_chart(fig_k, use_container_width=True, key=f"{chart_key_prefix}_candlestick")
            with fig_col2:
                fig_line = plot_line_chart(df_history, db['ticker'])
                st.plotly_chart(fig_line, use_container_width=True, key=f"{chart_key_prefix}_line")
        else:
            st.warning("⚠️ 無法取得歷史股價資料繪製圖表 (可能為大盤趨勢或標的代號錯誤)。")

    with tab2:
        raw_news = db['news_data'].get("raw_news", [])
        if not raw_news:
            st.warning("⚠️ 系統在過去 5 天內未能搜尋到相關新聞。")
        else:
            st.success(f"共抓取到 {len(raw_news)} 則新聞，平均情緒分數: **{s_score}**")
            st.dataframe(pd.DataFrame(raw_news), width="stretch", hide_index=True)
            
    with tab3:
        st.markdown("#### 量化模型核心參數與明細")
        qd = db['quant_data']
        
        main_metrics = {
            "預測報酬率 (Predicted Return)": f"{qd.get('predicted_return', 0)*100:.3f}%",
            "綜合訊號 (Signal)": qd.get('signal', '未知'),
            "市場狀態 (Regime)": qd.get('regime', '未知'),
            "歷史最大回撤 (Max DD)": qd.get('max_dd', '未知'),
            "最佳模型名稱 (Best Model)": qd.get('model_name', '未知')
        }
        
        st.dataframe(pd.DataFrame(list(main_metrics.items()), columns=["指標名稱", "數值"]), hide_index=True, use_container_width=True)
        
        if 'model_details' in qd and isinstance(qd['model_details'], list):
            st.markdown("##### 參與決策的底層模型明細")
            st.dataframe(pd.DataFrame(qd['model_details']), hide_index=True, use_container_width=True)
        else:
            with st.expander("檢視原始 JSON"):
                st.json(qd)
        
    st.divider()
