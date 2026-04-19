import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import streamlit.components.v1 as components

# --- 1. CONFIG & STYLE ---
st.set_page_config(page_title="Black Swan Event Prediction", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anuphan:wght@300;400;600&display=swap');
    html, body, [data-testid="stAppViewContainer"] { background-color: #F8FAFC !important; font-family: 'Anuphan', sans-serif; }
    .section-card { background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #E2E8F0; }
    .main-title { color: #0F172A; font-weight: 700; font-size: 2.2rem; margin-bottom: 10px; text-align: center; }
    .shock-btn { background-color: #FEF2F2; border: 2px solid #EF4444; padding: 10px; border-radius: 10px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- 2. CORE ENGINES ---
@st.cache_data(ttl=3600)
def fetch_data():
    tickers = {
        'NSE_India': '^NSEI', 'NYSE': '^NYA', 'SSE': '000001.SS', 'JPX': '^N225', 
        'Euronext': '^N100', 'LSE': '^FTSE', 'VIX': '^VIX', 'Gold': 'GC=F', 
        'Oil': 'BZ=F', 'Copper': 'HG=F', 'USDX': 'DX-Y.NYB', 'TNX': '^TNX', 'IRX': '^IRX', 'SP500': '^GSPC'
    }
    df = yf.download(list(tickers.values()), start="1975-01-01")['Close'].ffill().bfill()
    df = df.rename(columns={v: k for k, v in tickers.items()})
    
    # Normalized Data for Section 1 Graph
    price_cols = ['NSE_India', 'NYSE', 'SSE', 'JPX', 'Euronext', 'LSE', 'Gold', 'Oil']
    df_norm = df[price_cols].copy()
    for col in df_norm.columns:
        df_norm[col] = (df_norm[col] / df_norm[col].dropna().iloc[0]) * 100
        
    return df, df_norm

def estimate_risk_mc(stress, horizon=30):
    risk_factor = np.power(stress/0.0549, 1.15) if stress > 0.0549 else stress/0.0549
    draws = np.random.random((30000, horizon))
    return (np.any(draws < (1/5000 * risk_factor), axis=1).sum() / 30000) * 100

# --- 3. INITIALIZATION ---
df, df_norm = fetch_data()
latest = df.iloc[-1]
prev = df.iloc[-2]

# Calculate Real-time Stress
returns = df[['NYSE', 'SP500', 'Euronext', 'LSE', 'JPX', 'SSE', 'NSE_India']].pct_change().dropna()
k_real = returns.rolling(252).kurt().mean(axis=1).iloc[-1]
v_real = (returns.rolling(252).std().mean(axis=1) * np.sqrt(252)).iloc[-1]
c_real = returns.tail(60).corr().where(np.triu(np.ones((7,7)), k=1).astype(bool)).stack().mean()
y_real = (latest['TNX'] - latest['IRX'])
g_real = (latest['Gold'] / latest['Copper'])

stress_real = (v_real * 0.3389 + abs(y_real/100) * 0.2450 + c_real * 0.1463 + (k_real/15) * 0.1411)
p_real_today = estimate_risk_mc(stress_real)

# --- UI START ---
st.markdown('<div class="main-title">🦢 Black Swan Event Prediction Model</div>', unsafe_allow_html=True)

# --- SECTION 1: WATCHTOWER ---
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🔭 Section 1: Live Data Watchtower")
w1, w2, w3, w4, w5, w6 = st.columns(6)
w7, w8, w9, w10, w11, w12 = st.columns(6)

m_data = [
    ('NYSE', 'NYSE', w1), ('LSE', 'LSE', w2), ('JPX', 'Nikkei', w3), 
    ('NSE_India', 'Nifty 50', w4), ('SSE', 'SSE (China)', w5), ('Euronext', 'Euro 100', w6),
    ('Gold', 'Gold ($)', w7), ('Oil', 'Brent Oil', w8), ('Copper', 'Copper', w9),
    ('VIX', 'VIX Index', w10), ('TNX', '10Y Yield', w11), ('USDX', 'USD Index', w12)
]

for key, label, col in m_data:
    diff = ((latest[key]-prev[key])/prev[key]*100)
    col.metric(label, f"{latest[key]:,.2f}", f"{diff:+.2f}%")

with st.expander("📈 View Long-term Growth Comparison (Since 1975)"):
    fig_long = go.Figure()
    for col in df_norm.columns:
        fig_long.add_trace(go.Scatter(x=df_norm.index, y=df_norm[col], name=col, line=dict(width=1)))
    fig_long.update_layout(yaxis_type="log", height=400, template="plotly_white", margin=dict(t=10, b=10))
    st.plotly_chart(fig_long, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- SECTION 2: PREDICTION ---
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🔮 Section 2: Prediction of Black Swan Event")
c_left, c_right = st.columns([1, 1.2])

with c_left:
    risk_idx = np.clip(stress_real / 0.5 * 100, 2.5, 98.5)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = risk_idx,
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#1E293B"},
                 'steps': [{'range': [0, 35], 'color': "#BBF7D0"}, {'range': [35, 70], 'color': "#FEF08A"}, {'range': [70, 100], 'color': "#FECACA"}]}
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=0, b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

with c_right:
    if risk_idx < 35: status, s_col, desc = "NORMAL", "#10B981", "สภาวะตลาดปกติ ระบบมีความยืดหยุ่นสูง"
    elif risk_idx < 70: status, s_col, desc = "ELEVATED", "#F59E0B", "ความเสี่ยงเริ่มสะสม ระบบมีความเครียดเหนือค่าเฉลี่ย"
    else: status, s_col, desc = "CRITICAL", "#EF4444", "ระบบเปราะบางขั้นสูงสุด เสี่ยงต่อภาวะ Black Swan"
    
    st.markdown(f"<div style='background:{s_col}22; padding:15px; border-radius:10px; border-left:5px solid {s_col}'><b>Current Status: {status}</b><br>{desc}</div>", unsafe_allow_html=True)
    st.write("#### Quantitative Model Descriptor")
    st.caption("โมเดลวิเคราะห์ผ่าน 4 มิติสำคัญ: Panic (Vol), Fragility (Coupling), Fat-tail (Kurtosis) และ Macro (Yield Spread) โดยใช้ Monte Carlo Simulation 30,000 รอบ เพื่อคำนวณโอกาสเกิดเหตุการณ์ที่เป็น Extreme Outliers")
    
    st.metric("Probability Today (Real-time)", f"{p_real_today:.2f}%")

st.markdown('</div>', unsafe_allow_html=True)

# --- SECTION 3: SANDBOX ---
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🎮 Section 3: Simulation Probability Sandbox")

s1, s2, s3 = st.columns(3)
s_vol = s1.slider("Volatility (Panic)", 0.05, 0.90, float(v_real))
s_kurt = s1.slider("Kurtosis (Fat-Tail)", 0.0, 20.0, float(k_real))
s_yield = s2.slider("Yield Spread", -1.50, 1.50, float(y_real))
s_gold = s2.slider("Gold/Copper Ratio", 200.0, 1000.0, float(g_real))
s_coupling = s3.slider("Global Coupling", 0.0, 1.0, float(c_real))
st.markdown('<div class="shock-btn">', unsafe_allow_html=True)
butterfly = st.checkbox("🔥 ACTIVATE SURPRISE SHOCK (Butterfly Effect)")
st.markdown('</div>', unsafe_allow_html=True)

stress_sim = (s_vol * 0.3389 + abs(s_yield/100) * 0.2450 + s_coupling * 0.1463 + (s_kurt/15) * 0.1411) * (1.8 if butterfly else 1.0)
p_sim = estimate_risk_mc(stress_sim)

# Simulation UI
sc1, sc2 = st.columns([1, 2])
with sc1:
    if p_sim < 5: tid, lbl, t_col = "15568846810302620355", "Happy Duck", "#10B981"
    elif p_sim < 15: tid, lbl, t_col = "13982082229451252813", "Anxious Duck", "#F59E0B"
    else: tid, lbl, t_col = "25805348", "PANIC DUCK!", "#EF4444"
    
    components.html(f'<div class="tenor-gif-embed" data-postid="{tid}" data-share-method="host" data-aspect-ratio="1.0" data-width="100%"></div><script type="text/javascript" async src="https://tenor.com/embed.js"></script>', height=280)
    st.markdown(f"<h3 style='text-align:center; color:{t_col};'>{lbl}</h3>", unsafe_allow_html=True)

with sc2:
    st.metric("Simulated Risk Today", f"{p_sim:.2f}%", delta=f"{p_sim-p_real_today:+.2f}%", delta_color="inverse")
    
    # Sensitivity Matrix with "YOU ARE HERE" Star
    v_axis = np.linspace(max(0.05, s_vol-0.2), min(0.9, s_vol+0.2), 10)
    y_axis = np.linspace(s_yield-1.0, s_yield+1.0, 10)
    z = [[estimate_risk_mc((v*0.34 + abs(y/100)*0.25 + s_coupling*0.15)*(1.8 if butterfly else 1.0)) for v in v_axis] for y in y_axis]
    
    fig_heat = go.Figure(data=go.Heatmap(z=z, x=np.round(v_axis,2), y=np.round(y_axis,2), colorscale='RdYlGn_r'))
    fig_heat.add_trace(go.Scatter(x=[s_vol], y=[s_yield], mode='markers', marker=dict(color='white', size=15, symbol='star', line=dict(color='black', width=2)), name="YOU"))
    fig_heat.update_layout(height=350, title="Risk Sensitivity Map (Vol vs Yield)", margin=dict(t=30, b=0))
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
