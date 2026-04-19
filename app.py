import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import streamlit.components.v1 as components

# --- 1. APP CONFIG & STYLING ---
st.set_page_config(page_title="Black Swan Predictor", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anuphan:wght@300;400;600&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F1F5F9 !important;
        font-family: 'Anuphan', sans-serif;
    }
    .main-title { color: #1E293B; font-weight: 700; font-size: 2.8rem; margin-bottom: 0; }
    .section-card {
        background-color: white; padding: 25px; border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); margin-bottom: 20px;
    }
    .stMetric { background-color: #f8fafc; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; }
    .shake { animation: shake 0.5s infinite; }
    @keyframes shake {
        0% { transform: translate(1px, 1px) rotate(0deg); }
        10% { transform: translate(-1px, -2px) rotate(-1deg); }
        20% { transform: translate(-3px, 0px) rotate(1deg); }
        100% { transform: translate(1px, -2px) rotate(-1deg); }
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. BACKEND ENGINES ---
@st.cache_data(ttl=3600)
def fetch_all_data():
    tickers = {
        'NSE_India': '^NSEI', 'NYSE': '^NYA', 'SSE': '000001.SS', 'JPX': '^N225', 
        'Euronext': '^N100', 'LSE': '^FTSE', 'VIX': '^VIX', 'Gold': 'GC=F', 
        'Oil': 'BZ=F', 'Copper': 'HG=F', 'USDX': 'DX-Y.NYB', 'TNX': '^TNX', 'IRX': '^IRX'
    }
    df = yf.download(list(tickers.values()), start="2022-01-01")['Close'].ffill().bfill()
    df = df.rename(columns={v: k for k, v in tickers.items()})
    return df

def get_risk_metrics(df):
    markets = ['NYSE', 'Euronext', 'LSE', 'JPX', 'SSE', 'NSE_India']
    returns = df[markets].pct_change().dropna()
    kurt = returns.rolling(252).kurt().mean(axis=1).iloc[-1]
    vol = (returns.rolling(252).std().mean(axis=1) * np.sqrt(252)).iloc[-1]
    coupling = returns.tail(60).corr().where(np.triu(np.ones((6,6)), k=1).astype(bool)).stack().mean()
    yield_spread = (df['TNX'] - df['IRX']).iloc[-1]
    stress = (vol * 0.3389 + abs(yield_spread/100) * 0.2450 + coupling * 0.1463 + (kurt/15) * 0.1411)
    return stress, vol, kurt, yield_spread, coupling

def estimate_black_swan_mc(stress, horizon_days=30, simulations=30000):
    baseline = 1/5000
    risk_factor = np.power(stress/0.0549, 1.15) if stress > 0.0549 else stress/0.0549
    draws = np.random.random((simulations, horizon_days))
    return (np.any(draws < (baseline * risk_factor), axis=1).sum() / simulations) * 100

# --- 3. DATA INITIALIZATION ---
df = fetch_all_data()
stress_real, v_real, k_real, y_real, c_real = get_risk_metrics(df)
p_real_today = estimate_black_swan_mc(stress_real)
p_real_3m = estimate_black_swan_mc(stress_real + 0.012)
p_real_6m = estimate_black_swan_mc(stress_real + 0.025)

# --- SIDEBAR: METHODOLOGY ---
with st.sidebar:
    st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHRreXF4eXF4eXF4eXF4eXF4eXF4eXF4eXF4eXF4eXF4&ep=v1_internal_gif_by_id&id=V4p3fA80OQ5eU/giphy.gif", width=100)
    st.header("Methodology")
    st.info("""
    **Antifragile Risk Model v3**
    - Monte Carlo Simulations: 30,000+
    - Yield Spread: Recession Signal
    - Kurtosis: Fat-Tail measurement
    - Coupling: Systemic Fragility
    """)
    st.divider()
    st.caption("Data source: Yahoo Finance")

# --- APP TITLE ---
st.markdown('<div class="main-title">🦢 Black Swan Event Prediction Model</div>', unsafe_allow_html=True)
st.markdown("<p style='color: #64748B; font-size: 1.2rem;'>Quantitative Risk Watchtower & Stress Test Simulator</p>", unsafe_allow_html=True)

# --- SECTION 1: LIVE DATA WATCHTOWER ---
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🔭 Section 1: Live Data Watchtower")
w1, w2, w3, w4, w5, w6 = st.columns(6)
markets_list = [('NYSE', 'NYSE'), ('LSE', 'LSE'), ('JPX', 'Nikkei'), ('NSE_India', 'Nifty 50'), ('Gold', 'Gold'), ('VIX', 'VIX')]
for i, (key, name) in enumerate(markets_list):
    val = df[key].iloc[-1]
    prev = df[key].iloc[-2]
    diff = ((val-prev)/prev*100)
    [w1, w2, w3, w4, w5, w6][i].metric(name, f"{val:,.1f}", f"{diff:+.2f}%")
st.markdown('</div>', unsafe_allow_html=True)

# --- SECTION 2: PREDICTION OF BLACK SWAN EVENT ---
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🔮 Section 2: Prediction of Black Swan Event")
p1, p2 = st.columns([1, 1.5])

with p1:
    risk_score = np.clip(stress_real / 0.5 * 100, 0, 100)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = risk_score,
        title = {'text': "Systemic Risk Index", 'font': {'size': 20}},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#1E293B"},
                 'steps': [{'range': [0, 35], 'color': "#BBF7D0"}, {'range': [35, 70], 'color': "#FEF08A"}, {'range': [70, 100], 'color': "#FECACA"}]}
    ))
    fig_gauge.update_layout(height=350, margin=dict(t=50, b=0, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with p2:
    st.write("### Probability Forecast (Real-time)")
    f1, f2, f3 = st.columns(3)
    f1.metric("Today", f"{p_real_today:.2f}%")
    f2.metric("3M Forward", f"{p_real_3m:.2f}%")
    f3.metric("6M Forward", f"{p_real_6m:.2f}%")
    
    # Simple Area Chart for Forecast
    fig_forecast = go.Figure(go.Scatter(x=["Today", "3M", "6M"], y=[p_real_today, p_real_3m, p_real_6m], fill='tozeroy', line=dict(color='#0F172A')))
    fig_forecast.update_layout(height=200, margin=dict(t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_forecast, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- SECTION 3: SIMULATION PROBABILITY SANDBOX ---
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🎮 Section 3: Simulation Probability Sandbox")

# Control Panel
with st.expander("🛠️ Simulator Controls", expanded=True):
    s1, s2, s3 = st.columns(3)
    s_vol = s1.slider("Volatility (Panic)", 0.05, 0.90, float(v_real))
    s_kurt = s1.slider("Kurtosis (Fat-Tail)", 0.0, 20.0, float(k_real))
    s_yield = s2.slider("Yield Spread", -1.50, 1.50, float(y_real))
    s_coupling = s2.slider("Global Coupling", 0.0, 1.0, float(c_real))
    butterfly = s3.checkbox("🦋 Activate Surprise Shock")
    chaos_mult = 1.8 if butterfly else 1.0

# Sim Calculation
stress_sim = (s_vol * 0.3389 + abs(s_yield/100) * 0.2450 + s_coupling * 0.1463 + (s_kurt/15) * 0.1411) * chaos_mult
p_sim_today = estimate_black_swan_mc(stress_sim)
p_sim_3m = estimate_black_swan_mc(stress_sim + 0.015)

# Visual Logic
status_color = "#10B981" if p_sim_today < 5 else ("#F59E0B" if p_sim_today < 15 else "#EF4444")
shake_class = "shake" if p_sim_today >= 5 else ""

sc1, sc2 = st.columns([1, 2])
with sc1:
    st.markdown(f"<div class='{shake_class}'>", unsafe_allow_html=True)
    if p_sim_today < 5:
        tid, lbl = "15568846810302620355", "Happy Duck"
    elif p_sim_today < 15:
        tid, lbl = "13982082229451252813", "Anxious Duck"
    else:
        tid, lbl = "25805348", "PANIC DUCK!"
    
    components.html(f"""
        <div class="tenor-gif-embed" data-postid="{tid}" data-share-method="host" data-aspect-ratio="1.0" data-width="100%"></div>
        <script type="text/javascript" async src="https://tenor.com/embed.js"></script>
    """, height=300)
    st.markdown(f"<h3 style='text-align:center; color:{status_color};'>{lbl}</h3></div>", unsafe_allow_html=True)

with sc2:
    st.markdown(f"### Simulated Risk: <span style='color:{status_color};'>{p_sim_today:.2f}%</span>", unsafe_allow_html=True)
    sm1, sm2 = st.columns(2)
    sm1.metric("Sim Today", f"{p_sim_today:.2f}%", delta=f"{p_sim_today-p_real_today:+.2f}%", delta_color="inverse")
    sm2.metric("Sim 3M", f"{p_sim_3m:.2f}%", delta=f"{p_sim_3m-p_real_3m:+.2f}%", delta_color="inverse")

    # Heatmap
    vr = np.linspace(max(0.05, s_vol-0.2), min(0.9, s_vol+0.2), 8)
    yr = np.linspace(s_yield-1.0, s_yield+1.0, 8)
    z = [[estimate_black_swan_mc((v*0.34 + abs(y/100)*0.25 + s_coupling*0.15)*chaos_mult) for v in vr] for y in yr]
    
    fig_h = go.Figure(data=go.Heatmap(z=z, x=np.round(vr,2), y=np.round(yr,2), colorscale='RdYlGn_r'))
    fig_h.update_layout(height=300, title="Risk Sensitivity Matrix", margin=dict(t=30, b=0))
    st.plotly_chart(fig_h, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
