import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

  /* ── Global Reset ── */
  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #050a14;
    color: #e2e8f0;
  }
  .main { background: #050a14; }
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1220 0%, #070d1a 100%);
    border-right: 1px solid #1e2d48;
  }

  /* ── Hero Banner ── */
  .hero-banner {
    background: linear-gradient(135deg, #0a1628 0%, #0d2045 50%, #091830 100%);
    border: 1px solid #1a3a6e;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 80px rgba(0, 120, 255, 0.08);
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(0,100,255,0.1) 0%, transparent 70%);
    animation: pulse 4s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
  }
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
    line-height: 1.1;
  }
  .hero-sub {
    color: #64748b;
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    color: #60a5fa;
    padding: 0.3rem 0.9rem;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
  }

  /* ── Metric Cards ── */
  .metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
  .metric-card {
    flex: 1; min-width: 160px;
    background: #0b1628;
    border: 1px solid #1e2d48;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
  }
  .metric-card:hover { border-color: #3b82f6; }
  .metric-card .mc-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.5rem;
  }
  .metric-card .mc-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
  }
  .metric-card .mc-accent { color: #38bdf8; }
  .metric-card .mc-sub { font-size: 0.75rem; color: #334155; margin-top: 0.4rem; }

  /* ── Section Headers ── */
  .section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 3px solid #3b82f6;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
  }

  /* ── Input Card ── */
  .input-card {
    background: #0b1628;
    border: 1px solid #1e2d48;
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
  }
  .input-card h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    color: #94a3b8;
    margin: 0 0 1rem 0;
    font-weight: 700;
  }

  /* ── Result Panels ── */
  .result-safe {
    background: linear-gradient(135deg, #0a2818 0%, #0b2f1b 100%);
    border: 1px solid #15803d;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(21,128,61,0.15);
    animation: glow-green 2s ease-in-out infinite alternate;
  }
  @keyframes glow-green {
    from { box-shadow: 0 0 20px rgba(21,128,61,0.1); }
    to   { box-shadow: 0 0 50px rgba(21,128,61,0.25); }
  }
  .result-fraud {
    background: linear-gradient(135deg, #2a0a0a 0%, #300b0b 100%);
    border: 1px solid #dc2626;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(220,38,38,0.2);
    animation: glow-red 1.5s ease-in-out infinite alternate;
  }
  @keyframes glow-red {
    from { box-shadow: 0 0 20px rgba(220,38,38,0.15); }
    to   { box-shadow: 0 0 60px rgba(220,38,38,0.35); }
  }
  .result-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
  .result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 0.3rem 0;
  }
  .result-label.safe { color: #4ade80; }
  .result-label.fraud { color: #f87171; }
  .result-prob {
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 0.3rem;
  }

  /* ── Model Info Pills ── */
  .info-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(30,45,72,0.8);
    border: 1px solid #1e2d48;
    border-radius: 99px;
    padding: 0.35rem 0.85rem;
    font-size: 0.78rem;
    color: #94a3b8;
    margin: 0.25rem;
  }
  .info-pill span { color: #60a5fa; font-weight: 600; }

  /* ── Streamlit Overrides ── */
  .stSlider > div > div > div > div { background: #3b82f6 !important; }
  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.35) !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    box-shadow: 0 6px 28px rgba(59,130,246,0.5) !important;
    transform: translateY(-1px) !important;
  }
  div[data-testid="stNumberInput"] input,
  div[data-testid="stTextInput"] input {
    background: #0a1628 !important;
    border: 1px solid #1e2d48 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
  }
  .stSelectbox > div > div {
    background: #0a1628 !important;
    border: 1px solid #1e2d48 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
  }
  div[data-testid="stTab"] button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
  }
  .stDataFrame { border: 1px solid #1e2d48 !important; border-radius: 10px !important; }

  /* Sidebar */
  .sidebar-model-info {
    background: rgba(11,22,40,0.8);
    border: 1px solid #1e2d48;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
  }
  .sidebar-model-info h4 {
    font-family: 'Syne', sans-serif;
    color: #60a5fa;
    font-size: 0.9rem;
    margin: 0 0 0.8rem 0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .sidebar-row { display: flex; justify-content: space-between; padding: 0.3rem 0; border-bottom: 1px solid #0f1d33; font-size: 0.82rem; }
  .sidebar-row:last-child { border-bottom: none; }
  .sidebar-key { color: #475569; }
  .sidebar-val { color: #94a3b8; font-weight: 600; }

  /* Plotly chart dark bg fix */
  .js-plotly-plot .plotly, .js-plotly-plot .plotly div { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    scaler = joblib.load(r'C:\Users\HP\OneDrive\Desktop\DataScience\Data Science\Fraud_Detection_System\standard_scaler.joblib')
    model = joblib.load(r'C:\Users\HP\OneDrive\Desktop\DataScience\Data Science\Fraud_Detection_System\xgboost_fraud_detection_model.joblib')
    return scaler, model

scaler, model = load_models()

FEATURES     = list(model.feature_names_in_)
IMPORTANCES  = model.feature_importances_
TOP_FEATURES = sorted(zip(FEATURES, IMPORTANCES), key=lambda x: x[1], reverse=True)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.2rem 0;">
      <span style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800;
        background:linear-gradient(135deg,#60a5fa,#818cf8);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        🛡️ FraudGuard AI
      </span>
      <div style="font-size:0.72rem; color:#334155; margin-top:0.3rem; letter-spacing:0.08em; text-transform:uppercase;">
        Credit Card Fraud Detection
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Model Info
    st.markdown("""
    <div class="sidebar-model-info">
      <h4>🤖 XGBoost Model</h4>
      <div class="sidebar-row"><span class="sidebar-key">Type</span><span class="sidebar-val">XGBClassifier</span></div>
      <div class="sidebar-row"><span class="sidebar-key">Objective</span><span class="sidebar-val">binary:logistic</span></div>
      <div class="sidebar-row"><span class="sidebar-key">Features</span><span class="sidebar-val">31</span></div>
      <div class="sidebar-row"><span class="sidebar-key">Classes</span><span class="sidebar-val">Legit / Fraud</span></div>
      <div class="sidebar-row"><span class="sidebar-key">Random State</span><span class="sidebar-val">42</span></div>
    </div>
    <div class="sidebar-model-info">
      <h4>⚖️ Standard Scaler</h4>
      <div class="sidebar-row"><span class="sidebar-key">Feature</span><span class="sidebar-val">Amount</span></div>
      <div class="sidebar-row"><span class="sidebar-key">Mean (μ)</span><span class="sidebar-val">$88.64</span></div>
      <div class="sidebar-row"><span class="sidebar-key">Std (σ)</span><span class="sidebar-val">$250.20</span></div>
      <div class="sidebar-row"><span class="sidebar-key">Variance</span><span class="sidebar-val">62,599</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    nav = st.radio("Navigation", ["🔍 Detect Fraud", "📊 Model Analysis", "🧪 Batch Test"], label_visibility="collapsed")


# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">⚡ Real-time AI Detection</div>
  <div class="hero-title">FraudGuard AI</div>
  <p class="hero-sub">XGBoost-powered credit card fraud detection with PCA feature analysis &amp; real-time risk scoring</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DETECT FRAUD
# ═══════════════════════════════════════════════════════════════════════════════
if "🔍 Detect Fraud" in nav:

    st.markdown('<div class="section-header">Transaction Input</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="input-card"><h3>💳 Transaction Details</h3>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("Amount ($)", min_value=0.0, max_value=50000.0, value=150.0, step=0.01, format="%.2f")
            time_val = st.number_input("Time (seconds)", min_value=0.0, max_value=172800.0, value=3600.0, step=1.0)
        with c2:
            st.metric("Avg Transaction", "$88.64", help="Mean from scaler training data")
            st.metric("Std Dev", "$250.20", help="Standard deviation from scaler training data")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-card"><h3>📐 Top PCA Features</h3>', unsafe_allow_html=True)
        st.caption("V14, V10, V2 drive ~93% of model decisions")

        top3_vals = {}
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            top3_vals['V14'] = st.slider("V14 🔑", -20.0, 20.0, 0.0, 0.1, help="Most important feature (65.5%)")
        with tc2:
            top3_vals['V10'] = st.slider("V10 🔑", -20.0, 20.0, 0.0, 0.1, help="2nd most important (14.9%)")
        with tc3:
            top3_vals['V2'] = st.slider("V2 🔑", -20.0, 20.0, 0.0, 0.1, help="3rd most important (11.5%)")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-card"><h3>🔢 Other PCA Features (V1–V28)</h3>', unsafe_allow_html=True)
        other_vals = {}
        for feat in FEATURES:
            if feat not in ['Time', 'Amount', 'log_amount', 'V14', 'V10', 'V2']:
                other_vals[feat] = 0.0

        with st.expander("⚙️ Advanced: Adjust remaining PCA features"):
            cols_v = st.columns(4)
            i = 0
            for feat in FEATURES:
                if feat not in ['Time', 'Amount', 'log_amount', 'V14', 'V10', 'V2']:
                    with cols_v[i % 4]:
                        other_vals[feat] = st.number_input(feat, value=0.0, format="%.3f", key=f"feat_{feat}")
                    i += 1
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">Prediction Engine</div>', unsafe_allow_html=True)

        if st.button("🛡️  RUN FRAUD ANALYSIS", use_container_width=True):
            with st.spinner("Analyzing transaction..."):
                time.sleep(0.6)

            # Build input
            scaled_amount = (amount - scaler.mean_[0]) / scaler.scale_[0]
            log_amount    = np.log1p(amount)

            row = {}
            row['Time']       = time_val
            row['Amount']     = scaled_amount
            row['log_amount'] = log_amount
            row['V14']        = top3_vals['V14']
            row['V10']        = top3_vals['V10']
            row['V2']         = top3_vals['V2']
            for feat in FEATURES:
                if feat not in row:
                    row[feat] = other_vals.get(feat, 0.0)

            X = np.array([[row[f] for f in FEATURES]])
            proba = model.predict_proba(X)[0]
            pred  = model.predict(X)[0]
            fraud_prob = proba[1] * 100
            safe_prob  = proba[0] * 100

            # Result
            if pred == 0:
                st.markdown(f"""
                <div class="result-safe">
                  <div class="result-icon">✅</div>
                  <div class="result-label safe">LEGITIMATE</div>
                  <div class="result-prob">Confidence: {safe_prob:.3f}% safe &nbsp;|&nbsp; {fraud_prob:.4f}% fraud risk</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-fraud">
                  <div class="result-icon">🚨</div>
                  <div class="result-label fraud">FRAUD DETECTED</div>
                  <div class="result-prob">Fraud probability: {fraud_prob:.3f}% &nbsp;|&nbsp; Legitimate: {safe_prob:.4f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=fraud_prob,
                domain={'x': [0,1], 'y': [0,1]},
                title={'text': "Fraud Risk Score", 'font': {'size': 18, 'color': '#94a3b8', 'family': 'Space Grotesk'}},
                number={'suffix': '%', 'font': {'size': 36, 'color': '#f1f5f9', 'family': 'Syne'}},
                delta={'reference': 50, 'relative': False, 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#334155', 'tickfont': {'color': '#64748b'}},
                    'bar': {'color': '#dc2626' if fraud_prob > 50 else '#16a34a', 'thickness': 0.3},
                    'bgcolor': '#0b1628',
                    'bordercolor': '#1e2d48',
                    'steps': [
                        {'range': [0, 30],  'color': 'rgba(22,163,74,0.15)'},
                        {'range': [30, 70], 'color': 'rgba(234,179,8,0.1)'},
                        {'range': [70, 100],'color': 'rgba(220,38,38,0.15)'},
                    ],
                    'threshold': {'line': {'color': '#f59e0b', 'width': 3}, 'thickness': 0.85, 'value': 50}
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=260,
                margin=dict(t=30, b=0, l=20, r=20),
                font_color='#94a3b8',
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

            # Transaction summary
            st.markdown('<div class="section-header">Transaction Summary</div>', unsafe_allow_html=True)
            scol1, scol2, scol3 = st.columns(3)
            scol1.metric("Raw Amount", f"${amount:,.2f}")
            scol2.metric("Scaled Amount", f"{scaled_amount:.4f}")
            scol3.metric("Log(Amount+1)", f"{log_amount:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif "📊 Model Analysis" in nav:

    st.markdown('<div class="section-header">Model Intelligence Dashboard</div>', unsafe_allow_html=True)

    # Top stats
    top1_feat, top1_imp = TOP_FEATURES[0]
    top2_feat, top2_imp = TOP_FEATURES[1]
    top3_feat, top3_imp = TOP_FEATURES[2]

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="mc-label">🧠 Model Type</div>
        <div class="mc-value mc-accent" style="font-size:1.2rem; padding-top:0.3rem;">XGBoost</div>
        <div class="mc-sub">binary:logistic</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">📐 Total Features</div>
        <div class="mc-value">31</div>
        <div class="mc-sub">1 Time · 28 PCA · Amount · Log</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">🏆 Top Feature</div>
        <div class="mc-value mc-accent">{top1_feat}</div>
        <div class="mc-sub">{top1_imp*100:.1f}% importance</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">🥈 2nd Feature</div>
        <div class="mc-value mc-accent">{top2_feat}</div>
        <div class="mc-sub">{top2_imp*100:.1f}% importance</div>
      </div>
      <div class="metric-card">
        <div class="mc-label">🥉 3rd Feature</div>
        <div class="mc-value mc-accent">{top3_feat}</div>
        <div class="mc-sub">{top3_imp*100:.1f}% importance</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "⚖️ Scaler Analysis", "🔬 Model Architecture"])

    # ── TAB 1: Feature Importance ──────────────────────────────────────────────
    with tab1:
        names_sorted = [f for f,_ in TOP_FEATURES]
        imps_sorted  = [i for _,i in TOP_FEATURES]
        colors_bar   = ['#dc2626' if i==imps_sorted[0] else '#2563eb' if i==imps_sorted[1] else '#0891b2' if i==imps_sorted[2] else '#1e3a5f' for i in imps_sorted]

        fig_fi = go.Figure()
        fig_fi.add_trace(go.Bar(
            x=imps_sorted,
            y=names_sorted,
            orientation='h',
            marker=dict(color=colors_bar, line=dict(width=0)),
            text=[f"{v*100:.2f}%" for v in imps_sorted],
            textposition='outside',
            textfont=dict(color='#64748b', size=11),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        fig_fi.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=700,
            margin=dict(t=20, b=20, l=10, r=80),
            yaxis=dict(autorange='reversed', gridcolor='#0f1d33', tickfont=dict(color='#64748b', size=11)),
            xaxis=dict(gridcolor='#0f1d33', tickfont=dict(color='#64748b'), tickformat='.2%'),
            showlegend=False,
            font=dict(family='Space Grotesk'),
        )
        st.plotly_chart(fig_fi, use_container_width=True, config={'displayModeBar': False})

        # Treemap
        st.markdown('<div class="section-header">Feature Importance Treemap</div>', unsafe_allow_html=True)
        fig_tree = go.Figure(go.Treemap(
            labels=names_sorted,
            parents=[''] * len(names_sorted),
            values=imps_sorted,
            textinfo='label+percent root',
            marker=dict(
                colors=imps_sorted,
                colorscale=[[0,'#0f1d33'],[0.3,'#1d4ed8'],[0.7,'#2563eb'],[1,'#dc2626']],
                showscale=True,
                colorbar=dict(tickfont=dict(color='#64748b'), outlinewidth=0),
            ),
            hovertemplate='<b>%{label}</b><br>Importance: %{value:.6f}<br>%{percentRoot} of total<extra></extra>',
        ))
        fig_tree.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(t=10, b=10, l=10, r=10),
            font=dict(family='Space Grotesk', color='#e2e8f0'),
        )
        st.plotly_chart(fig_tree, use_container_width=True, config={'displayModeBar': False})

    # ── TAB 2: Scaler Analysis ─────────────────────────────────────────────────
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div style="background:#0b1628;border:1px solid #1e2d48;border-radius:14px;padding:1.5rem;margin-bottom:1rem;">
              <div style="font-family:'Syne',sans-serif;font-size:1rem;color:#60a5fa;margin-bottom:1rem;font-weight:700;">STANDARD SCALER — Amount Feature</div>
              <div class="sidebar-row"><span class="sidebar-key">Feature Scaled</span><span class="sidebar-val">Amount</span></div>
              <div class="sidebar-row"><span class="sidebar-key">Mean (μ)</span><span class="sidebar-val">$88.6367</span></div>
              <div class="sidebar-row"><span class="sidebar-key">Standard Dev (σ)</span><span class="sidebar-val">$250.1986</span></div>
              <div class="sidebar-row"><span class="sidebar-key">Variance (σ²)</span><span class="sidebar-val">62,599.35</span></div>
              <div class="sidebar-row" style="border-bottom:none;"><span class="sidebar-key">Formula</span><span class="sidebar-val">(x − μ) / σ</span></div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            amounts = np.linspace(0, 1000, 300)
            scaled  = (amounts - scaler.mean_[0]) / scaler.scale_[0]
            fig_scaler = go.Figure()
            fig_scaler.add_trace(go.Scatter(x=amounts, y=scaled, mode='lines',
                line=dict(color='#3b82f6', width=2.5),
                fill='tozeroy', fillcolor='rgba(59,130,246,0.08)',
                name='Scaled Amount',
                hovertemplate='$%{x:.2f} → %{y:.4f}<extra></extra>'))
            fig_scaler.add_hline(y=0, line_dash='dot', line_color='#334155', annotation_text="Mean=$88.64")
            fig_scaler.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=260, title=dict(text="Amount → Scaled Value", font=dict(color='#94a3b8', size=13, family='Space Grotesk')),
                margin=dict(t=40,b=20,l=10,r=10),
                xaxis=dict(title="Original Amount ($)", gridcolor='#0f1d33', tickfont=dict(color='#64748b')),
                yaxis=dict(title="Scaled Value", gridcolor='#0f1d33', tickfont=dict(color='#64748b')),
                font=dict(family='Space Grotesk'),
            )
            st.plotly_chart(fig_scaler, use_container_width=True, config={'displayModeBar': False})

        # Interactive scaler demo
        st.markdown('<div class="section-header">Interactive Scaler Demo</div>', unsafe_allow_html=True)
        demo_amount = st.slider("Test Amount ($)", 0.0, 5000.0, 150.0, 1.0)
        demo_scaled = (demo_amount - scaler.mean_[0]) / scaler.scale_[0]
        demo_log    = np.log1p(demo_amount)
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Raw Amount", f"${demo_amount:,.2f}")
        sc2.metric("Scaled Amount", f"{demo_scaled:.6f}", delta=f"{'above' if demo_scaled>0 else 'below'} mean")
        sc3.metric("Log1p(Amount)", f"{demo_log:.6f}")

    # ── TAB 3: Architecture ────────────────────────────────────────────────────
    with tab3:
        st.markdown("""
        <div class="input-card">
        <h3>🏗️ Pipeline Architecture</h3>
        """, unsafe_allow_html=True)

        pipeline_steps = [
            ("📥", "Raw Transaction", "Time, Amount, V1–V28"),
            ("⚖️", "Standard Scaler", "Normalizes Amount feature\n(μ=88.64, σ=250.20)"),
            ("🔢", "Feature Engineering", "Adds log_amount = log(1 + Amount)"),
            ("🤖", "XGBoost Classifier", "31 features → binary:logistic\nOutputs P(fraud) ∈ [0,1]"),
            ("🎯", "Decision", "Threshold @ 0.5\n→ Legitimate or Fraud"),
        ]

        cols_p = st.columns(len(pipeline_steps))
        for col, (icon, title, desc) in zip(cols_p, pipeline_steps):
            with col:
                st.markdown(f"""
                <div style="background:#0a1628;border:1px solid #1e2d48;border-radius:12px;padding:1rem;text-align:center;height:160px;display:flex;flex-direction:column;justify-content:center;">
                  <div style="font-size:1.8rem;margin-bottom:0.4rem;">{icon}</div>
                  <div style="font-family:'Syne',sans-serif;font-size:0.85rem;color:#60a5fa;font-weight:700;margin-bottom:0.4rem;">{title}</div>
                  <div style="font-size:0.72rem;color:#475569;line-height:1.4;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Feature table
        st.markdown('<div class="section-header">All 31 Features — Ranked by Importance</div>', unsafe_allow_html=True)
        df_feats = pd.DataFrame({
            'Rank':       range(1, len(TOP_FEATURES)+1),
            'Feature':    [f for f,_ in TOP_FEATURES],
            'Importance': [round(i, 6) for _,i in TOP_FEATURES],
            'Importance %': [f"{i*100:.4f}%" for _,i in TOP_FEATURES],
            'Category':   ['PCA' if f.startswith('V') else 'Raw' if f in ['Time','Amount'] else 'Engineered' for f,_ in TOP_FEATURES],
        })
        st.dataframe(df_feats, use_container_width=True, hide_index=True, height=400)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH TEST
# ═══════════════════════════════════════════════════════════════════════════════
elif "🧪 Batch Test" in nav:

    st.markdown('<div class="section-header">Batch Transaction Simulator</div>', unsafe_allow_html=True)
    st.caption("Generate synthetic transactions and run them through the fraud detection pipeline.")

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        n_txns = st.slider("Number of Transactions", 10, 500, 100)
    with bc2:
        fraud_pct = st.slider("Simulated Fraud % (V14 bias)", 0, 50, 5)
    with bc3:
        seed = st.number_input("Random Seed", value=42, step=1)

    if st.button("⚡  RUN BATCH SIMULATION", use_container_width=True):
        with st.spinner(f"Simulating {n_txns} transactions..."):
            rng = np.random.RandomState(int(seed))
            rows = []
            for i in range(n_txns):
                is_fraud_sim = rng.random() < (fraud_pct / 100)
                row = {'Time': rng.uniform(0, 172800), 'Amount': abs(rng.exponential(88.64))}
                for feat in FEATURES:
                    if feat.startswith('V'):
                        if feat == 'V14' and is_fraud_sim:
                            row[feat] = rng.uniform(-15, -5)
                        elif feat == 'V10' and is_fraud_sim:
                            row[feat] = rng.uniform(-8, -2)
                        else:
                            row[feat] = rng.normal(0, 1)
                row['Amount_raw'] = row['Amount']
                row['Amount']     = (row['Amount'] - scaler.mean_[0]) / scaler.scale_[0]
                row['log_amount'] = np.log1p(row['Amount_raw'])
                rows.append(row)

            df_batch = pd.DataFrame(rows)
            X_batch  = df_batch[[f for f in FEATURES]].values
            probas   = model.predict_proba(X_batch)
            preds    = model.predict(X_batch)

            df_batch['fraud_prob']  = probas[:, 1]
            df_batch['prediction']  = ['🚨 Fraud' if p==1 else '✅ Legit' for p in preds]
            df_batch['risk_level']  = df_batch['fraud_prob'].apply(
                lambda p: '🔴 High' if p > 0.7 else ('🟡 Medium' if p > 0.3 else '🟢 Low'))
            df_batch['amount_disp'] = df_batch['Amount_raw'].apply(lambda a: f"${a:,.2f}")

        n_fraud = (preds==1).sum()
        n_legit = (preds==0).sum()

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="mc-label">Total Transactions</div>
            <div class="mc-value">{n_txns}</div>
          </div>
          <div class="metric-card">
            <div class="mc-label">✅ Legitimate</div>
            <div class="mc-value" style="color:#4ade80;">{n_legit}</div>
            <div class="mc-sub">{n_legit/n_txns*100:.1f}%</div>
          </div>
          <div class="metric-card">
            <div class="mc-label">🚨 Fraud Detected</div>
            <div class="mc-value" style="color:#f87171;">{n_fraud}</div>
            <div class="mc-sub">{n_fraud/n_txns*100:.1f}%</div>
          </div>
          <div class="metric-card">
            <div class="mc-label">Avg Fraud Score</div>
            <div class="mc-value mc-accent">{probas[:,1].mean()*100:.2f}%</div>
          </div>
          <div class="metric-card">
            <div class="mc-label">Max Fraud Score</div>
            <div class="mc-value" style="color:#f87171;">{probas[:,1].max()*100:.2f}%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        ch1, ch2 = st.columns(2)

        with ch1:
            fig_pie = go.Figure(go.Pie(
                labels=['Legitimate', 'Fraud'],
                values=[n_legit, n_fraud],
                hole=0.65,
                marker=dict(colors=['#16a34a', '#dc2626']),
                textfont=dict(family='Space Grotesk', color='#e2e8f0'),
                hovertemplate='<b>%{label}</b>: %{value} (%{percent})<extra></extra>',
            ))
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=300, margin=dict(t=10,b=10,l=10,r=10),
                showlegend=True,
                legend=dict(font=dict(color='#94a3b8', family='Space Grotesk'), bgcolor='rgba(0,0,0,0)'),
                annotations=[dict(text=f"<b>{n_txns}</b><br>Total", x=0.5, y=0.5, font_size=16, showarrow=False, font_color='#94a3b8', font_family='Space Grotesk')],
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

        with ch2:
            fig_hist = go.Figure(go.Histogram(
                                          x=probas[:,1],
                                          nbinsx=40,
                                          marker=dict(
                                              color=probas[:,1],
                                              colorscale=[[0,'#1d4ed8'],[1,'#dc2626']],
                                              cmin=0,
                                              cmax=1,
                                              opacity=0.85,
                                              line=dict(color='#1e3a5f', width=0.5),
                                          ),
            hovertemplate='Score: %{x:.3f}<br>Count: %{y}<extra></extra>',
                                      ))
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                xaxis=dict(title="Fraud Probability Score", gridcolor='#0f1d33', tickfont=dict(color='#64748b'), tickformat='.0%'),
                yaxis=dict(title="Count", gridcolor='#0f1d33', tickfont=dict(color='#64748b')),
                margin=dict(t=10,b=10,l=10,r=10),
                font=dict(family='Space Grotesk'),
            )
            st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

        # Results table
        st.markdown('<div class="section-header">Transaction Results</div>', unsafe_allow_html=True)
        display_df = df_batch[['amount_disp','Time','fraud_prob','prediction','risk_level']].copy()
        display_df.columns = ['Amount','Time (s)','Fraud Score','Prediction','Risk Level']
        display_df['Fraud Score'] = display_df['Fraud Score'].apply(lambda x: f"{x*100:.4f}%")
        display_df['Time (s)']   = display_df['Time (s)'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem 0; color:#1e2d48; font-size:0.78rem; letter-spacing:0.05em;">
  FraudGuard AI &nbsp;·&nbsp; XGBoost + Standard Scaler Pipeline &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
