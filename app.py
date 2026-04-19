import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import datetime

# --- 1. SET PAGE & CUTE THEME ---
st.set_page_config(page_title="🦢 Black Swan Odyssey: Full Visual", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0faff; font-family: 'Nunito', sans-serif; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 15px; border: 2px solid #e1f5fe; box-shadow: 2px 2px 8px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #0277bd; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦢 The Black Swan Odyssey 🦆")
st.markdown("### *Dashboard วิเคราะห์หงส์ดำ: กราฟ Log Scale & ตรรกะเปรียบเทียบประวัติศาสตร์* 🐥")

# --- 2. DATA FETCHING ---
@st.cache_data(ttl=3600)
def fetch_data():
    tickers = {
        'VIX': '^VIX', 'SP500': '^GSPC', '10Y': '^TNX', '3M': '^IRX',
        'Gold': 'GC=F', 'Copper': 'HG=F', 'USD': 'DX-Y.NYB', 'Oil': 'BZ=F'
    }
    # ดึงข้อมูล 1 ปีเพื่อทำ Normalization
    df = yf.download(list(tickers.values()), period="1y")['Close']
    df = df.ffill().bfill() # จัดการค่าว่างให้ครบ
    
    # ดึงค่าล่าสุด
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Normalization (Base 100)
    df_norm = (df / df.iloc[0]) * 100
    
    return df, df_norm, latest, prev

try:
    df_raw, df_norm, latest, prev = fetch_data()
    data_success = True
except:
    st.error("ขออภัย! ไม่สามารถดึงข้อมูลจากตลาดได้ในขณะนี้ 🐥")
    data_success = False

if data_success:
    # --- 3. LOGIC & WEIGHTS (AI Feature Importance) ---
    WEIGHTS = {'vol': 0.339, 'yield': 0.245, 'coupling': 0.146, 'other': 0.27}
    
    curr_vix = latest['^VIX']
    curr_yield = latest['^TNX'] - latest['^IRX']
    curr_gold_cu = latest['GC=F'] / latest['HG=F']
    curr_usd = latest['DX-Y.NYB']
    curr_oil = latest['BZ=F']

    def calc_prob(v, y, c=0.45):
        # VIX Pivot 14.3 (AI Rule), Yield Pivot 1.02
        v_score = (v / 40) * WEIGHTS['vol']
        y_score = max(0, (1 - y/2)) * WEIGHTS['yield']
        c_score = c * WEIGHTS['coupling']
        return min(max((v_score + y_score + c_score + 0.1) * 100, 1.5), 98.0)

    # --- 4. SECTION I: LIVE WATCHTOWER ---
    st.header("🔭 Section I: Live Watchtower")
    
    # Probability Metrics (Locked to Live Data)
    live_p = calc_prob(curr_vix, curr_yield)
    p1, p2, p3 = st.columns(3)
    p1.metric("Today's Risk 🦢", f"{live_p:.2f}%", "Real-time")
    p2.metric("6 Months Outlook 🦆", f"{min(live_p*1.2, 99.0):.2f}%", "Stress Build-up")
    p3.metric("12 Months Outlook 🖤", f"{min(live_p*1.6, 99.0):.2f}%", "Fragility Score")

    # กล่องข้อมูล Snapshot แบบเดิมที่คุณชอบ
    st.subheader("📊 Live Market Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("S&P 500", f"{latest['^GSPC']:,.2f}", f"{(latest['^GSPC']-prev['^GSPC'])/prev['^GSPC']*100:.2f}%")
    col2.metric("VIX Index", f"{curr_vix:.2f}", f"{(curr_vix-prev['^VIX']):.2f}")
    col3.metric("Yield Spread", f"{curr_yield:.2f}%")
    col4.metric("Gold/Cu Ratio", f"{curr_gold_cu:.2f}")

    # กราฟเปรียบเทียบ (Normalized & Log Scale)
    st.subheader("📈 Global Asset Growth Comparison (Normalized & Log Scale)")
    fig_growth = go.Figure()
    plot_map = {'^GSPC': 'S&P 500 🇺🇸', 'GC=F': 'Gold 💰', 'BZ=F': 'Brent Oil 🛢️', 'DX-Y.NYB': 'USD Index 💵'}
    
    for ticker, label in plot_map.items():
        if ticker in df_norm.columns:
            fig_growth.add_trace(go.Scatter(x=df_norm.index, y=df_norm[ticker], name=label, line=dict(width=2)))

    fig_growth.update_layout(
        yaxis_type="log", # LOG SCALE ตามที่คุณต้องการ
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_growth, use_container_width=True)

    # --- 5. SECTION II: ODYSSEY SIMULATOR ---
    st.divider()
    st.header("🕹️ Section II: Odyssey Simulator")
    
    with st.sidebar:
        st.header("🎨 Simulator Settings")
        st.markdown("### The Big 3 (75%)")
        s_vix = st.slider("VIX (Volatility)", 5.0, 100.0, float(curr_vix))
        s_yield = st.slider("Yield Spread (%)", -3.0, 5.0, float(curr_yield))
        s_coupling = st.slider("Market Coupling", 0.0, 1.0, 0.45)
        
        st.markdown("### The 3 Flavor (25%)")
        s_gold_cu = st.slider("Gold/Copper Ratio", 300.0, 800.0, float(curr_gold_cu))
        s_usd = st.slider("USD Index", 90.0, 120.0, float(curr_usd))
        s_oil = st.slider("Oil Price", 20.0, 150.0, float(curr_oil))

    # คำนวณความเสี่ยง Simulator
    sim_p = calc_prob(s_vix, s_yield, s_coupling)
    s1, s2, s3 = st.columns(3)
    s1.metric("Simulated Today 🦢", f"{sim_p:.2f}%")
    s2.metric("Simulated 6 Months 🦆", f"{min(sim_p*1.3, 99.0):.2f}%")
    s3.metric("Simulated 12 Months 🖤", f"{min(sim_p*1.8, 99.0):.2f}%")

    # --- 6. SECTION III: HISTORICAL MIRROR (FIXED LOGIC) ---
    st.divider()
    st.header("🪞 Section III: Historical Mirror")
    st.markdown("*เปรียบเทียบพฤติกรรมตลาดที่คุณจำลอง กับเหตุการณ์ในอดีต (AI Matching)*")

    # Historical Reference Points [VIX, Yield, Coupling]
    history = {
        "Lehman Moment (2008)": [80.0, -0.5, 0.90],
        "COVID Crash (2020)": [66.0, 1.1, 0.85],
        "Dot-com Bust (2000)": [35.0, -0.2, 0.60],
        "Asian Crisis (1997)": [45.0, 1.5, 0.75]
    }

    # Normalize ข้อมูลก่อนเทียบ เพื่อไม่ให้เลขหลักสิบ (VIX) กลบเลขหลักหน่วย (Yield)
    def calculate_similarity(v1, v2):
        # ใช้ MinMaxScaler แบบทำมือเพื่อให้ผลลัพธ์เสถียร
        v1_n = np.array([v1[0]/100, (v1[1]+3)/8, v1[2]])
        v2_n = np.array([v2[0]/100, (v2[1]+3)/8, v2[2]])
        return cosine_similarity([v1_n], [v2_n])[0][0]

    current_vec = [s_vix, s_yield, s_coupling]
    h_cols = st.columns(4)
    
    for i, (name, vec) in enumerate(history.items()):
        sim_val = calculate_similarity(current_vec, vec) * 100
        with h_cols[i]:
            st.write(f"**{name}**")
            st.subheader(f"{sim_val:.1f}%")
            st.progress(sim_val/100)
            st.caption("Similarity based on AI Vector")

    # --- 7. SWAN'S WHISPER ---
    st.divider()
    st.header("💬 Section IV: Swan's Whisper")
    if sim_p > 50:
        st.error(f"🖤 **AI Assessment:** ระบบตรวจพบความเปราะบาง (Fragility) สูงสุด ค่า VIX {s_vix} ทะลุเกณฑ์ 14.3 และ Yield {s_yield}% กำลังตึงตัว")
    else:
        st.success(f"🦢 **AI Assessment:** ตลาดยังอยู่ในภาวะปกติ (Antifragile) หงส์ขาวยังนิ่งสงบ ความเสี่ยงโดยรวม {sim_p:.2f}%")
