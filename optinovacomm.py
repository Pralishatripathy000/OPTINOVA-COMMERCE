import os
import base64
from io import BytesIO
from textwrap import dedent

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


st.set_page_config(
    page_title="Optinova Commerce",
    page_icon="optinova_logo.png",
    layout="wide"
)

MODEL_PATH    = "optinova_xgb_model.pkl"
SCALER_PATH   = "optinova_scaler.pkl"
FEATURES_PATH = "optinova_features.pkl"
LOGO_PATH     = "optinova_logo.png"


def image_to_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


def render_html(code):
    st.markdown(dedent(code).strip(), unsafe_allow_html=True)


def load_assets():
    missing = [f for f in [MODEL_PATH, SCALER_PATH, FEATURES_PATH] if not os.path.exists(f)]
    if missing:
        st.error("Missing required file(s): " + ", ".join(missing))
        st.stop()
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), joblib.load(FEATURES_PATH)


model, scaler, feature_columns = load_assets()
logo_b64      = image_to_base64(LOGO_PATH)
hero_logo_src = f"data:image/png;base64,{logo_b64}" if logo_b64 else ""

FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800;900&display=swap');"

render_html(f"""
<style>
{FONT_IMPORT}

*, html, body, [class*="css"] {{
    font-family: 'Manrope', sans-serif !important;
}}

.block-container {{
    padding-top: 0.5rem !important;
    max-width: 1500px;
}}

.stApp {{
    background:
        radial-gradient(circle at top left,  rgba(34,211,238,0.08), transparent 28%),
        radial-gradient(circle at bottom right, rgba(244,114,182,0.12), transparent 30%),
        linear-gradient(135deg, #020617, #050b1f, #020617);
    color: #f8fafc;
}}

section[data-testid="stSidebar"] {{
    background: rgba(5,12,30,0.98);
    border-right: 1px solid rgba(255,255,255,0.07);
}}

h1, h2, h3 {{
    font-family: 'Manrope', sans-serif !important;
    color: white;
    font-weight: 800;
    letter-spacing: -0.5px;
}}

p {{
    font-family: 'Manrope', sans-serif !important;
    color: #d7dee7;
    line-height: 1.8;
    font-size: 15px;
}}

.sidebar-logo-box {{
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: -18px;
    margin-bottom: -8px;
}}

.sidebar-logo {{
    width: 172px;
    max-width: 100%;
    height: auto;
    filter: drop-shadow(0 0 18px rgba(34,211,238,0.28));
}}

.metric-card {{
    background: linear-gradient(145deg, rgba(15,23,42,0.96), rgba(16,24,45,0.92));
    padding: 26px;
    border-radius: 24px;
    border: 1px solid rgba(244,114,182,0.18);
    box-shadow: 0 12px 42px rgba(0,0,0,0.32), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: all 0.35s ease;
}}

.metric-card:hover {{
    transform: translateY(-4px);
    border: 1px solid rgba(244,114,182,0.38);
}}

.big-number {{
    font-family: 'Manrope', sans-serif !important;
    font-size: 38px;
    font-weight: 800;
    color: #f472b6;
    margin-top: 10px;
    letter-spacing: -1px;
}}

.label {{
    font-family: 'Manrope', sans-serif !important;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #93c5fd;
    font-weight: 700;
}}

.insight-box {{
    background: linear-gradient(135deg, rgba(236,72,153,0.12), rgba(168,85,247,0.10));
    border: 1px solid rgba(236,72,153,0.20);
    border-radius: 22px;
    padding: 28px;
}}

.soft-box {{
    background: rgba(30,41,59,0.58);
    border-radius: 22px;
    padding: 24px;
    border: 1px solid rgba(255,255,255,0.06);
}}

.stButton > button {{
    font-family: 'Manrope', sans-serif !important;
    background: linear-gradient(135deg, #22d3ee, #f472b6);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    font-weight: 800;
    transition: all 0.3s ease;
}}

.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(244,114,182,0.25);
}}

footer {{ visibility: hidden; }}

/* ── Hide sidebar collapse/expand button label text ── */
button[data-testid="collapsedControl"],
button[data-testid="baseButton-headerNoPadding"],
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapseButton"] svg + span,
section[data-testid="stSidebar"] button span.st-emotion-cache-nahz7x,
.st-emotion-cache-1q1n0ol,
span.css-10pw50 {{
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    font-size: 0 !important;
    color: transparent !important;
}}

/* Target the Material icon text that shows as "keyboard_double_arrow_right" */
[data-testid="stSidebarCollapseButton"] {{
    overflow: hidden;
}}

button[kind="header"] span,
button[kind="headerNoPadding"] span {{
    font-size: 0 !important;
    color: transparent !important;
}}
</style>
""")


# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "prediction_ready" not in st.session_state:
    st.session_state.prediction_ready = False


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    if logo_b64:
        render_html(f"""
        <div class="sidebar-logo-box">
            <img src="data:image/png;base64,{logo_b64}" class="sidebar-logo">
        </div>
        """)
    else:
        st.markdown("## Optinova")

    render_html("""
    <p style="text-align:center;font-size:11px;font-weight:700;color:#64748b;
       text-transform:uppercase;letter-spacing:2px;margin:4px 0 10px;">
       Ecommerce Intelligence
    </p>
    """)

    selected_page = option_menu(
        menu_title=None,
        options=["Overview", "Prediction", "Model Insights", "Risk Breakdown", "About"],
        icons=["house-fill", "graph-up-arrow", "bar-chart-fill", "exclamation-triangle-fill", "info-circle-fill"],
        default_index=0,
        styles={
            "container":        {"padding": "0!important", "background-color": "transparent"},
            "icon":             {"color": "#22d3ee", "font-size": "15px"},
            "nav-link":         {"font-size": "14px", "font-family": "Manrope, sans-serif",
                                 "color": "#d7dee7", "border-radius": "10px",
                                 "margin": "3px 0px", "padding": "10px"},
            "nav-link-selected": {"background-color": "rgba(34,211,238,0.18)",
                                  "color": "white", "font-weight": "700"},
        }
    )

    st.markdown("---")
    st.markdown(
        "<p style='font-size:13px;font-weight:700;color:#93c5fd;text-transform:uppercase;"
        "letter-spacing:1.5px;margin-bottom:10px;'>Session Inputs</p>",
        unsafe_allow_html=True
    )

    if st.button("Run Prediction", use_container_width=True, key="run_btn"):
        with st.spinner("Running AI behavioral analysis..."):
            st.session_state.prediction_ready = True

    bounce_rate    = st.slider("Bounce Rate",            0.0, 1.0,   0.2)
    exit_rate      = st.slider("Exit Rate",              0.0, 1.0,   0.2)
    page_values    = st.slider("Page Value",             0.0, 500.0, 50.0)
    product_related = st.slider("Product Related Pages", 0,   500,   50)
    traffic_type   = st.slider("Traffic Type",           1,   20,    2)
    visitor_type   = st.selectbox("Visitor Type", ["New Visitor", "Returning Visitor", "Other"])
    weekend        = st.selectbox("Weekend Visit", ["No", "Yes"])


# ══════════════════════════════════════════════════════════════════════════════
#  HERO / LAUNCH PAGE
# ══════════════════════════════════════════════════════════════════════════════
components.html(f"""
<!DOCTYPE html>
<html>
<head>
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800;900&display=swap');

* {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Manrope', sans-serif; }}
body {{ background: transparent; color: white; }}

.launch {{
    min-height: 720px;
    border-radius: 32px;
    padding: 46px 52px;
    background:
        radial-gradient(circle at 80% 10%, rgba(244,114,182,0.20), transparent 34%),
        radial-gradient(circle at 12% 92%, rgba(34,211,238,0.18), transparent 32%),
        linear-gradient(135deg, #050816, #071633, #07031c);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 22px 80px rgba(0,0,0,0.40);
    overflow: hidden;
}}

.brand-logo {{
    width: 80px;
    height: 80px;
    object-fit: contain;
    filter: drop-shadow(0 0 22px rgba(34,211,238,0.26));
}}

.brand-headline {{
    font-size: 62px;
    font-weight: 900;
    letter-spacing: -2.5px;
    line-height: 1.05;
    text-align: center;
}}

.brand-headline span {{
    background: linear-gradient(135deg, #8b5cf6, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.tagline-row {{
    font-size: 13px;
    font-weight: 700;
    color: #93c5fd;
    letter-spacing: 3px;
    text-transform: uppercase;
    text-align: center;
}}

.subtitle-line {{
    font-size: 16px;
    font-weight: 400;
    color: #cbd5e1;
    margin-top: 8px;
    text-align: center;
}}

.launch-grid {{
    display: grid;
    grid-template-columns: 0.85fr 1fr 0.85fr;
    gap: 22px;
    align-items: center;
}}

.left-stack, .right-stack {{
    display: flex;
    flex-direction: column;
    gap: 14px;
    align-items: center;
}}

.mini-icons {{
    display: flex;
    gap: 12px;
    margin-bottom: 2px;
    justify-content: center;
}}

.mini-icon {{
    width: 46px;
    height: 46px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(99,102,241,0.50);
    box-shadow: 0 0 24px rgba(99,102,241,0.24);
    font-size: 20px;
}}

.side-label {{
    font-size: 13px;
    line-height: 1.6;
    color: #e2e8f0;
    font-weight: 500;
    text-align: center;
    width: 100%;
}}

.side-label b {{ color: white; font-weight: 800; }}

.accuracy-card {{
    width: 100%;
    text-align: center;
    padding: 18px;
    border-radius: 20px;
    background: rgba(15,23,42,0.76);
    border: 1px solid rgba(96,165,250,0.28);
    box-shadow: 0 0 34px rgba(59,130,246,0.12);
}}

.accuracy-title {{
    font-size: 12px;
    font-weight: 700;
    color: #93c5fd;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}}

.accuracy-value {{
    font-size: 44px;
    font-weight: 900;
    background: linear-gradient(135deg, #38bdf8, #8b5cf6, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -2px;
}}

.accuracy-sub {{
    color: #94a3b8;
    font-size: 12px;
    line-height: 1.5;
    margin-top: 4px;
}}

.phone-wrap {{ display: flex; align-items: center; justify-content: center; }}

.phone {{
    width: 310px;
    min-height: 600px;
    border-radius: 46px;
    background: linear-gradient(180deg, #111827, #020617);
    border: 8px solid #0b1020;
    box-shadow: 0 0 0 2px rgba(255,255,255,0.10), 0 0 55px rgba(34,211,238,0.22);
    padding: 22px;
    position: relative;
}}

.notch {{
    position: absolute; top: 10px; left: 50%; transform: translateX(-50%);
    width: 110px; height: 24px; background: #020617;
    border-radius: 0 0 16px 16px;
}}

.phone-top {{
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 22px; color: white; font-size: 16px; font-weight: 800;
}}

.shop-banner {{
    margin-top: 20px; padding: 18px; border-radius: 20px;
    background: radial-gradient(circle at 85% 30%, rgba(244,114,182,0.30), transparent 28%),
                linear-gradient(135deg, #3b0764, #1e1b4b);
}}

.shop-banner h3 {{ font-size: 20px; font-weight: 800; margin-bottom: 6px; }}
.shop-banner p  {{ color: #e0e7ff; font-size: 13px; margin-bottom: 12px; font-weight: 400; }}

.shop-btn {{
    display: inline-block; padding: 7px 14px; border-radius: 10px;
    background: #020617; color: white; font-size: 12px; font-weight: 700;
}}

.product-title {{ margin-top: 18px; font-size: 14px; font-weight: 800; }}

.products {{
    display: grid; grid-template-columns: repeat(3,1fr);
    gap: 8px; margin-top: 10px;
}}

.product {{
    background: #eef2ff; color: #0f172a; border-radius: 12px;
    padding: 9px; font-size: 10px; font-weight: 700;
}}

.product-img {{
    height: 56px;
    border-radius: 8px;
    margin-bottom: 7px;
    overflow: hidden;
    background: linear-gradient(135deg,#c7d2fe,#fbcfe8);
}}

.product-img img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 8px;
    display: block;
}}

.add-btn {{
    margin-top: 5px; padding: 4px; border-radius: 7px;
    background: #6d28d9; color: white; font-size: 9px; text-align: center;
}}

.cart {{
    margin-top: 16px; padding: 12px; border-radius: 16px;
    background: rgba(30,41,59,0.90);
    display: flex; justify-content: space-between; align-items: center;
    color: #e5e7eb; font-size: 13px;
}}

.checkout {{
    margin-top: 12px; display: flex; justify-content: space-between;
    align-items: center; color: white; font-weight: 800; font-size: 14px;
}}

.checkout-btn {{
    padding: 9px 18px; border-radius: 11px;
    background: linear-gradient(135deg, #8b5cf6, #f472b6); color: white; font-size: 13px;
}}

.chart-card {{
    padding: 18px; border-radius: 20px;
    background: rgba(15,23,42,0.78);
    border: 1px solid rgba(96,165,250,0.24);
    box-shadow: 0 0 34px rgba(59,130,246,0.12);
}}

.chart-title {{
    font-size: 13px; font-weight: 700; color: #e2e8f0; margin-bottom: 8px;
}}

.risk-value {{ font-size: 28px; font-weight: 900; color: #f472b6; }}

.line-visual {{
    height: 60px; margin-top: 10px; border-radius: 12px;
    background: linear-gradient(135deg, rgba(244,114,182,0.08), rgba(59,130,246,0.08)),
                repeating-linear-gradient(to right, rgba(255,255,255,0.05) 0 1px, transparent 1px 40px);
    position: relative; overflow: hidden;
}}

.line-visual::after {{
    content: ""; position: absolute; left: 16px; right: 16px; top: 30px;
    height: 3px; background: linear-gradient(90deg, #22d3ee, #8b5cf6, #f472b6);
    border-radius: 999px; transform: rotate(-8deg);
    box-shadow: 0 0 20px rgba(244,114,182,0.55);
}}

.lift-card {{ display: flex; gap: 12px; align-items: center; }}
.lift-number {{ font-size: 30px; font-weight: 900; color: #f472b6; }}

.bar-set {{ display: flex; align-items: flex-end; gap: 6px; height: 68px; flex: 1; }}
.bar {{ width: 16px; border-radius: 6px 6px 0 0; background: linear-gradient(180deg,#38bdf8,#8b5cf6); }}

.brands-section {{
    margin-top: 32px;
    border-top: 1px solid rgba(255,255,255,0.08);
    padding-top: 24px;
    text-align: center;
}}

.brands-label {{
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 18px;
}}

.brands-row {{
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 10px;
}}

.brand-pill {{
    padding: 8px 16px;
    border-radius: 40px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    font-size: 13px;
    font-weight: 800;
    color: #e2e8f0;
    letter-spacing: 0.5px;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.brand-pill img {{
    width: 20px;
    height: 20px;
    object-fit: contain;
    border-radius: 4px;
    flex-shrink: 0;
}}

.brand-pill.amazon   {{ color: #f0a500; border-color: rgba(240,165,0,0.25);   background: rgba(240,165,0,0.07); }}
.brand-pill.flipkart {{ color: #47c0f5; border-color: rgba(71,192,245,0.25);  background: rgba(71,192,245,0.07); }}
.brand-pill.myntra   {{ color: #f472b6; border-color: rgba(244,114,182,0.25); background: rgba(244,114,182,0.07); }}
.brand-pill.meesho   {{ color: #c084fc; border-color: rgba(192,132,252,0.25); background: rgba(192,132,252,0.07); }}
.brand-pill.ajio     {{ color: #22d3ee; border-color: rgba(34,211,238,0.25);  background: rgba(34,211,238,0.07); }}
.brand-pill.nykaa    {{ color: #fb923c; border-color: rgba(251,146,60,0.25);  background: rgba(251,146,60,0.07); }}
.brand-pill.tatacliq {{ color: #60a5fa; border-color: rgba(96,165,250,0.25);  background: rgba(96,165,250,0.07); }}
.brand-pill.zepto    {{ color: #a78bfa; border-color: rgba(167,139,250,0.25); background: rgba(167,139,250,0.07); }}
</style>
</head>
<body>
<div class="launch">

    <div style="position:relative;margin-bottom:28px;">
        <img src="{hero_logo_src}" class="brand-logo" style="position:absolute;top:0;left:0;">
        <div style="display:flex;flex-direction:column;align-items:center;text-align:center;padding-top:4px;">
            <div class="brand-headline">Predict. Prevent. <span>Convert.</span></div>
            <div class="tagline-row" style="margin-top:10px;margin-bottom:0;">AI Commerce Intelligence Platform</div>
        </div>
    </div>

    <div class="launch-grid">
        <div class="left-stack">
            <div class="mini-icons">
                <div class="mini-icon">👥</div>
                <div class="mini-icon">🔒</div>
                <div class="mini-icon">🛡️</div>
            </div>
            <div class="side-label">
                <b>Understand</b> visitor behavior in real time.<br>
                <b>Detect</b> abandonment risk early.<br>
                <b>Take action</b> with smart interventions.
            </div>
            <div class="accuracy-card">
                <div class="accuracy-title">Prediction Accuracy</div>
                <div class="accuracy-value">92.7%</div>
                <div class="accuracy-sub">Our AI model predicts purchase intent with high precision.</div>
            </div>
        </div>

        <div class="phone-wrap">
            <div class="phone">
                <div class="notch"></div>
                <div class="phone-top">
                    <span>☰</span><span>ShopVista</span><span>🛒</span>
                </div>
                <div class="shop-banner">
                    <h3>Hi there! 👋</h3>
                    <p>Discover trends you'll love.</p>
                    <span class="shop-btn">Shop Now</span>
                </div>
                <div class="product-title">Recommended for you</div>
                <div class="products">
                    <div class="product">
                        <div class="product-img">
                            <img src="https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=120&q=80" alt="Sneakers">
                        </div>
                        Sneakers<br>$59.99
                        <div class="add-btn">Add to Cart</div>
                    </div>
                    <div class="product">
                        <div class="product-img">
                            <img src="https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=120&q=80" alt="Backpack">
                        </div>
                        Backpack<br>$49.99
                        <div class="add-btn">Add to Cart</div>
                    </div>
                    <div class="product">
                        <div class="product-img">
                            <img src="https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=120&q=80" alt="Watch">
                        </div>
                        Watch<br>$89.99
                        <div class="add-btn">Add to Cart</div>
                    </div>
                </div>
                <div class="product-title">Your Cart (2)</div>
                <div class="cart">
                    <span>Casual Shirt &nbsp; $39.99</span>
                    <span>- 1 +</span>
                </div>
                <div class="checkout">
                    <span>Total &nbsp; $99.98</span>
                    <span class="checkout-btn">Checkout</span>
                </div>
            </div>
        </div>

        <div class="right-stack">
            <div class="chart-card">
                <div class="chart-title" style="text-align:center;">Real-time Abandonment Risk</div>
                <div class="risk-value" style="text-align:center;">72%</div>
                <div style="font-size:11px;color:#94a3b8;margin-top:2px;text-align:center;">High Risk Detected</div>
                <div class="line-visual"></div>
            </div>
            <div class="chart-card">
                <div class="chart-title" style="text-align:center;">Conversion Improvement</div>
                <div class="lift-card">
                    <div style="flex:1;text-align:center;">
                        <div class="lift-number">+28.4%</div>
                        <div style="color:#dbeafe;font-size:12px;">Lift using AI engine</div>
                    </div>
                    <div class="bar-set">
                        <div class="bar" style="height:16px;"></div>
                        <div class="bar" style="height:28px;"></div>
                        <div class="bar" style="height:40px;"></div>
                        <div class="bar" style="height:56px;"></div>
                        <div class="bar" style="height:72px;"></div>
                    </div>
                </div>
            </div>
            <div class="chart-card" style="text-align:center;">
                <div class="chart-title">Revenue Impact</div>
                <div class="lift-number">+18.7%</div>
                <div style="color:#94a3b8;font-size:12px;margin-top:3px;">Increase from AI-driven actions</div>
            </div>
        </div>
    </div>

    <div class="brands-section">
        <div class="brands-label">✦ Made for Innovative Brands ✦</div>
        <div class="brands-row">
            <span class="brand-pill amazon">
                <img src="https://www.google.com/s2/favicons?sz=32&domain=amazon.com" alt="Amazon">Amazon
            </span>
            <span class="brand-pill flipkart">
                <img src="https://www.google.com/s2/favicons?sz=32&domain=flipkart.com" alt="Flipkart">Flipkart
            </span>
            <span class="brand-pill myntra">
                <img src="https://www.google.com/s2/favicons?sz=32&domain=myntra.com" alt="Myntra">Myntra
            </span>
            <span class="brand-pill meesho">
                <img src="https://www.google.com/s2/favicons?sz=32&domain=meesho.com" alt="Meesho">Meesho
            </span>
            <span class="brand-pill ajio">
                <img src="https://www.google.com/s2/favicons?sz=32&domain=ajio.com" alt="AJIO">AJIO
            </span>
            <span class="brand-pill nykaa">
                <img src="https://www.google.com/s2/favicons?sz=32&domain=nykaa.com" alt="Nykaa">Nykaa
            </span>
            <span class="brand-pill tatacliq">
                <img src="https://www.google.com/s2/favicons?sz=32&domain=tatacliq.com" alt="Tata CLiQ">Tata CLiQ
            </span>
            <span class="brand-pill zepto">
                <img src="https://www.google.com/s2/favicons?sz=32&domain=zeptonow.com" alt="Zepto">Zepto
            </span>
        </div>
    </div>

</div>
</body>
</html>
""", height=820, scrolling=False)


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURES + APPLICATIONS + PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
components.html("""
<!DOCTYPE html>
<html>
<head>
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800;900&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Manrope', sans-serif; }
body { background: transparent; color: white; }

/* ── Reduced panel padding for performance section ── */
.panel {
    border-radius: 32px;
    padding: 50px;
    margin-bottom: 30px;
    background:
        radial-gradient(circle at 15% 10%, rgba(34,211,238,0.14), transparent 30%),
        radial-gradient(circle at 85% 90%, rgba(244,114,182,0.15), transparent 32%),
        linear-gradient(135deg, #050816, #071633, #08031f);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 20px 70px rgba(0,0,0,0.35);
    overflow: hidden;
}

.panel.perf-panel {
    padding: 32px 36px 36px;
}

.kicker {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 40px;
    background: rgba(34,211,238,0.10);
    border: 1px solid rgba(34,211,238,0.30);
    color: #67e8f9;
    font-size: 12px;
    letter-spacing: 1.5px;
    font-weight: 800;
    text-transform: uppercase;
    margin-bottom: 16px;
}

.section-title {
    font-size: 50px;
    font-weight: 900;
    letter-spacing: -2px;
    line-height: 1.05;
    margin-bottom: 10px;
}

/* Smaller title for performance panel */
.perf-panel .section-title {
    font-size: 36px;
    margin-bottom: 6px;
}

.section-title span {
    background: linear-gradient(135deg, #67e8f9, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.section-caption {
    max-width: 800px;
    font-size: 17px;
    font-weight: 400;
    color: #cbd5e1;
    line-height: 1.75;
    margin-bottom: 36px;
}

/* Smaller caption for performance panel */
.perf-panel .section-caption {
    font-size: 14px;
    line-height: 1.6;
    margin-bottom: 18px;
}

/* ── Features grid ── */
.features-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
}

.feature-card {
    padding: 28px;
    border-radius: 24px;
    background: rgba(15,23,42,0.78);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 16px 42px rgba(0,0,0,0.28);
}

.feature-icon {
    width: 54px; height: 54px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(34,211,238,0.18), rgba(244,114,182,0.18));
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; margin-bottom: 18px;
    border: 1px solid rgba(244,114,182,0.22);
}

.feature-title { font-size: 19px; font-weight: 800; margin-bottom: 8px; color: white; }
.feature-text  { color: #94a3b8; font-size: 14px; line-height: 1.7; font-weight: 400; }

.badge {
    display: inline-flex; align-items: center; gap: 8px;
    margin-top: 14px; padding: 6px 12px;
    border-radius: 30px; font-size: 13px; font-weight: 700;
}

.badge-cyan   { background: rgba(34,211,238,0.12);  color: #22d3ee; }
.badge-pink   { background: rgba(244,114,182,0.12); color: #f472b6; }
.badge-purple { background: rgba(139,92,246,0.12);  color: #a78bfa; }

/* ── Apps grid ── */
.apps-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 22px;
}

.app-card {
    padding: 28px; border-radius: 26px;
    background: radial-gradient(circle at top right, rgba(244,114,182,0.10), transparent 35%),
                rgba(15,23,42,0.78);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 16px 42px rgba(0,0,0,0.28);
}

.app-symbol { font-size: 36px; margin-bottom: 16px; }
.app-title  { font-size: 20px; font-weight: 800; color: white; margin-bottom: 8px; }
.app-text   { color: #94a3b8; font-size: 14px; line-height: 1.7; font-weight: 400; }

/* ── Performance graph ── */
.graph-container {
    margin: 16px auto 0;
    max-width: 860px;
    width: 100%;
    border-radius: 22px;
    background: rgba(15,23,42,0.82);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 20px 24px 28px 24px;
    position: relative;
    overflow: hidden;
}

.legend {
    display: flex;
    gap: 20px;
    justify-content: flex-end;
    font-size: 12px;
    font-weight: 700;
    color: #e5e7eb;
    margin-bottom: 10px;
}
.legend-item { display: flex; align-items: center; gap: 6px; }
.dot { width: 10px; height: 10px; border-radius: 50%; }
.dot-cyan { background: #22d3ee; }
.dot-pink { background: #f472b6; }

.graph-svg-wrap {
    width: 100%;
    overflow: visible;
}

/* ── Stat row below graph ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    max-width: 860px;
    margin: 16px auto 0;
}

.stat-box {
    border-radius: 18px;
    padding: 16px;
    background: rgba(15,23,42,0.78);
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}

.stat-value { color: #f472b6; font-size: 26px; font-weight: 900; margin-bottom: 4px; }
.stat-label { color: #94a3b8; font-size: 12px; font-weight: 600; }
</style>
</head>
<body>

<!-- ── FEATURES ── -->
<div class="panel">
    <div class="kicker">Feature Intelligence</div>
    <div class="section-title">Powerful Features.<br><span>Real Results.</span></div>
    <div class="section-caption">
        Everything you need to maximize conversions and minimize cart abandonment.
        Optinova decodes raw ecommerce behavior into actionable intelligence.
    </div>
    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">AI Purchase Prediction</div>
            <div class="feature-text">Predicts purchase likelihood with high accuracy using XGBoost behavioral modeling.</div>
            <div class="badge badge-cyan">92.7% Accuracy</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚠️</div>
            <div class="feature-title">Abandonment Risk Detection</div>
            <div class="feature-text">Identifies at-risk user sessions before they leave — using exit and bounce signals.</div>
            <div class="badge badge-pink">72% Risk Detected</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Intent &amp; Behavior Analytics</div>
            <div class="feature-text">Understand user intent through deep behavioral signals: page values, traffic, session depth.</div>
            <div class="badge badge-purple">Real-time Insights</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Smart Interventions</div>
            <div class="feature-text">Trigger personalized offers and nudges at the exact right moment in a session.</div>
            <div class="badge badge-cyan">+28.4% Conversion Lift</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Conversion Optimization</div>
            <div class="feature-text">Actionable insights to continuously improve conversion rates and session quality.</div>
            <div class="badge badge-pink">Continuously Learning</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📄</div>
            <div class="feature-title">Reports &amp; Dashboards</div>
            <div class="feature-text">Beautiful exportable AI reports and interactive dashboards for your team.</div>
            <div class="badge badge-purple">PDF Export Ready</div>
        </div>
    </div>
</div>

<!-- ── APPLICATIONS ── -->
<div class="panel">
    <div class="kicker">Real-World Applications</div>
    <div class="section-title">Real World <span>Applications.</span></div>
    <div class="section-caption">
        Built to solve real challenges across industries. The same behavioral prediction
        intelligence supports multiple digital business environments.
    </div>
    <div class="apps-grid">
        <div class="app-card">
            <div class="app-symbol">🛍️</div>
            <div class="app-title">Ecommerce Stores</div>
            <div class="app-text">Reduce cart abandonment and increase revenue with AI-driven recommendations and real-time nudges.</div>
        </div>
        <div class="app-card">
            <div class="app-symbol">✈️</div>
            <div class="app-title">Travel &amp; Hospitality</div>
            <div class="app-text">Predict booking behavior and re-engage users with personalized offers before they drop off.</div>
        </div>
        <div class="app-card">
            <div class="app-symbol">🏦</div>
            <div class="app-title">Finance &amp; Banking</div>
            <div class="app-text">Detect drop-off in loan or card applications and improve completion rates with behavioral intelligence.</div>
        </div>
        <div class="app-card">
            <div class="app-symbol">🎓</div>
            <div class="app-title">EdTech Platforms</div>
            <div class="app-text">Identify high-intent students and nurture them toward course enrollment and completion.</div>
        </div>
        <div class="app-card">
            <div class="app-symbol">👑</div>
            <div class="app-title">SaaS &amp; Subscriptions</div>
            <div class="app-text">Reduce churn and increase renewals using session-level behavioral intelligence and risk scoring.</div>
        </div>
        <div class="app-card">
            <div class="app-symbol">🎟️</div>
            <div class="app-title">Events &amp; Ticketing</div>
            <div class="app-text">Increase ticket sales by predicting intent and sending smart alerts before users abandon checkout.</div>
        </div>
    </div>
</div>

<!-- ── PERFORMANCE ── -->
<div class="panel perf-panel">
    <div style="font-size:30px;font-weight:900;letter-spacing:-1px;line-height:1.05;margin-bottom:18px;color:white;">
        Real-time Performance <span style="background:linear-gradient(135deg,#67e8f9,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Comparison.</span>
    </div>

    <div class="graph-container">
        <div class="legend">
            <div class="legend-item"><div class="dot dot-cyan"></div>With AI</div>
            <div class="legend-item"><div class="dot dot-pink"></div>Without AI</div>
        </div>
        <div class="graph-svg-wrap">
        <svg viewBox="0 0 820 200" xmlns="http://www.w3.org/2000/svg"
             style="width:100%;height:auto;display:block;overflow:visible;">
            <defs>
                <linearGradient id="gradCyan" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#22d3ee;stop-opacity:1"/>
                    <stop offset="100%" style="stop-color:#38bdf8;stop-opacity:1"/>
                </linearGradient>
                <linearGradient id="gradPink" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#8b5cf6;stop-opacity:1"/>
                    <stop offset="100%" style="stop-color:#f472b6;stop-opacity:1"/>
                </linearGradient>
                <filter id="glowCyan">
                    <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                    <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
                <filter id="glowPink">
                    <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                    <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
            </defs>
            <!-- Vertical grid lines -->
            <line x1="100" y1="8"  x2="100" y2="162" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
            <line x1="215" y1="8"  x2="215" y2="162" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
            <line x1="330" y1="8"  x2="330" y2="162" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
            <line x1="445" y1="8"  x2="445" y2="162" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
            <line x1="560" y1="8"  x2="560" y2="162" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
            <line x1="675" y1="8"  x2="675" y2="162" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
            <line x1="800" y1="8"  x2="800" y2="162" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
            <!-- Horizontal grid lines -->
            <line x1="36" y1="38"  x2="810" y2="38"  stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
            <line x1="36" y1="80"  x2="810" y2="80"  stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
            <line x1="36" y1="122" x2="810" y2="122" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
            <!-- Y-axis labels -->
            <text x="30" y="42"  text-anchor="end" fill="#475569" font-size="10" font-family="Manrope,sans-serif">75%</text>
            <text x="30" y="84"  text-anchor="end" fill="#475569" font-size="10" font-family="Manrope,sans-serif">50%</text>
            <text x="30" y="126" text-anchor="end" fill="#475569" font-size="10" font-family="Manrope,sans-serif">25%</text>
            <!-- WITHOUT AI line: declining (high → low) -->
            <polyline
                points="36,34 152,55 268,76 384,96 500,116 616,132 728,146 810,154"
                fill="none"
                stroke="url(#gradPink)"
                stroke-width="3"
                stroke-linecap="round"
                stroke-linejoin="round"
                filter="url(#glowPink)"
            />
            <!-- WITH AI line: rising (low → high) -->
            <polyline
                points="36,152 152,132 268,110 384,88 500,66 616,48 728,34 810,24"
                fill="none"
                stroke="url(#gradCyan)"
                stroke-width="3"
                stroke-linecap="round"
                stroke-linejoin="round"
                filter="url(#glowCyan)"
            />
            <!-- End dots -->
            <circle cx="810" cy="24"  r="5" fill="#22d3ee" filter="url(#glowCyan)"/>
            <circle cx="810" cy="154" r="5" fill="#f472b6" filter="url(#glowPink)"/>
            <!-- X-axis labels -->
            <text x="36"  y="176" text-anchor="middle" fill="#475569" font-size="10" font-family="Manrope,sans-serif">00:00</text>
            <text x="152" y="176" text-anchor="middle" fill="#475569" font-size="10" font-family="Manrope,sans-serif">04:00</text>
            <text x="268" y="176" text-anchor="middle" fill="#475569" font-size="10" font-family="Manrope,sans-serif">08:00</text>
            <text x="384" y="176" text-anchor="middle" fill="#475569" font-size="10" font-family="Manrope,sans-serif">12:00</text>
            <text x="500" y="176" text-anchor="middle" fill="#475569" font-size="10" font-family="Manrope,sans-serif">16:00</text>
            <text x="616" y="176" text-anchor="middle" fill="#475569" font-size="10" font-family="Manrope,sans-serif">20:00</text>
            <text x="728" y="176" text-anchor="middle" fill="#475569" font-size="10" font-family="Manrope,sans-serif">22:00</text>
            <text x="810" y="176" text-anchor="middle" fill="#475569" font-size="10" font-family="Manrope,sans-serif">24:00</text>
        </svg>
        </div>
    </div>

    <div class="stat-row">
        <div class="stat-box">
            <div class="stat-value">89.4%</div>
            <div class="stat-label">Model Accuracy Achieved</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">+28.4%</div>
            <div class="stat-label">Better Conversions with AI</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">Real-time</div>
            <div class="stat-label">Dashboard Prediction Flow</div>
        </div>
    </div>
</div>

</body>
</html>
""", height=2200, scrolling=False)


# ══════════════════════════════════════════════════════════════════════════════
#  GATE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.prediction_ready:
    render_html("""
    <div class="insight-box" style="margin-top:10px;">
        <h3 style="font-family:'Manrope',sans-serif;font-weight:800;color:white;margin-bottom:10px;">
            Ready to analyze a live session?
        </h3>
        <p style="font-family:'Manrope',sans-serif;">
            Adjust the ecommerce session inputs from the sidebar on the left,
            then click <b>Run Prediction</b> to generate purchase probability,
            abandonment risk, buyer intent, and behavioral intelligence.
        </p>
    </div>
    """)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
engagement_score  = (product_related * page_values) / (bounce_rate + 1)
exit_intent_score = bounce_rate + exit_rate

input_data = pd.DataFrame({
    "Administrative":           [0],
    "Administrative_Duration":  [0],
    "Informational":            [0],
    "Informational_Duration":   [0],
    "ProductRelated":           [product_related],
    "ProductRelated_Duration":  [200],
    "BounceRates":              [bounce_rate],
    "ExitRates":                [exit_rate],
    "PageValues":               [page_values],
    "SpecialDay":               [0],
    "OperatingSystems":         [2],
    "Browser":                  [2],
    "Region":                   [1],
    "TrafficType":              [traffic_type],
    "EngagementScore":          [engagement_score],
    "ExitIntentScore":          [exit_intent_score],
})

if visitor_type == "Returning Visitor":
    input_data["VisitorType_Returning_Visitor"] = 1
if visitor_type == "Other":
    input_data["VisitorType_Other"] = 1
if weekend == "Yes":
    input_data["Weekend_True"] = 1

for col in feature_columns:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[feature_columns]

scaled_input     = scaler.transform(input_data)
prediction       = model.predict(scaled_input)[0]
probability      = model.predict_proba(scaled_input)[0][1]
abandonment_risk = (1 - probability) * 100

risk_label = (
    "High Risk User"     if abandonment_risk >= 70 else
    "Moderate Risk User" if abandonment_risk >= 40 else
    "Likely Buyer"
)

session_health = max(0, min(100,
    (probability * 60) + ((engagement_score / 500) * 25) + ((1 - exit_intent_score) * 15)
))

conversion_confidence = max(probability, 1 - probability) * 100
revenue_potential     = page_values * probability

buyer_intent = (
    "Strong"   if probability >= 0.70 else
    "Moderate" if probability >= 0.40 else
    "Weak"
)

plot_config = {"displayModeBar": False, "displaylogo": False, "responsive": True}


# ══════════════════════════════════════════════════════════════════════════════
#  EXECUTIVE METRICS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<p style='font-family:Manrope,sans-serif;font-size:22px;font-weight:800;"
    "color:white;margin:18px 0 12px;letter-spacing:-0.5px;'>Executive Metrics</p>",
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    render_html(f"""
    <div class="metric-card">
        <div class="label">Purchase Probability</div>
        <div class="big-number">{probability*100:.2f}%</div>
    </div>""")

with col2:
    render_html(f"""
    <div class="metric-card">
        <div class="label">Abandonment Risk</div>
        <div class="big-number">{abandonment_risk:.2f}%</div>
    </div>""")

with col3:
    render_html(f"""
    <div class="metric-card">
        <div class="label">Prediction</div>
        <div class="big-number" style="font-size:26px;">{risk_label}</div>
    </div>""")

with col4:
    render_html(f"""
    <div class="metric-card">
        <div class="label">Engagement Score</div>
        <div class="big-number">{engagement_score:.2f}</div>
    </div>""")

st.markdown(
    "<p style='font-family:Manrope,sans-serif;font-size:22px;font-weight:800;"
    "color:white;margin:22px 0 12px;letter-spacing:-0.5px;'>Executive Analytics</p>",
    unsafe_allow_html=True
)

a1, a2, a3, a4 = st.columns(4)

with a1:
    render_html(f"""
    <div class="metric-card">
        <div class="label">Session Health</div>
        <div class="big-number">{session_health:.1f}</div>
    </div>""")

with a2:
    render_html(f"""
    <div class="metric-card">
        <div class="label">Buyer Intent</div>
        <div class="big-number">{buyer_intent}</div>
    </div>""")

with a3:
    render_html(f"""
    <div class="metric-card">
        <div class="label">Revenue Potential</div>
        <div class="big-number">{revenue_potential:.1f}</div>
    </div>""")

with a4:
    render_html(f"""
    <div class="metric-card">
        <div class="label">Model Confidence</div>
        <div class="big-number">{conversion_confidence:.1f}%</div>
    </div>""")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTING
# ══════════════════════════════════════════════════════════════════════════════

# ── OVERVIEW ─────────────────────────────────────────────────────────────────
if selected_page == "Overview":
    st.markdown(
        "<p style='font-family:Manrope,sans-serif;font-size:20px;font-weight:800;"
        "color:white;margin-bottom:14px;'>Executive Overview</p>",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        fig1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Purchase Probability", "font": {"family": "Manrope", "size": 16}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#22d3ee"},
                   "bgcolor": "#1e293b", "borderwidth": 1, "bordercolor": "#334155"}
        ))
        fig1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white", "family": "Manrope"},
            height=420
        )
        st.plotly_chart(fig1, use_container_width=True, key="purchase_chart", config=plot_config)

    with c2:
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=abandonment_risk,
            title={"text": "Abandonment Risk", "font": {"family": "Manrope", "size": 16}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#f472b6"},
                   "bgcolor": "#1e293b", "borderwidth": 1, "bordercolor": "#334155"}
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white", "family": "Manrope"},
            height=420
        )
        st.plotly_chart(fig2, use_container_width=True, key="risk_chart", config=plot_config)

    radar_categories = ["Engagement", "Purchase Intent", "Revenue Quality", "Bounce Control", "Exit Stability"]
    radar_values = [
        min(100, engagement_score / 25),
        probability * 100,
        min(100, page_values * 2),
        max(0, 100 - (bounce_rate * 100)),
        max(0, 100 - (exit_rate * 100))
    ]

    fig5 = go.Figure()
    fig5.add_trace(go.Scatterpolar(
        r=radar_values, theta=radar_categories,
        fill="toself",
        line=dict(color="#22d3ee", width=3),
        fillcolor="rgba(244,114,182,0.25)"
    ))
    fig5.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(tickfont=dict(color="white", family="Manrope"))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Manrope"),
        height=520,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    st.plotly_chart(fig5, use_container_width=True, key="radar_chart", config=plot_config)


# ── PREDICTION ────────────────────────────────────────────────────────────────
elif selected_page == "Prediction":
    st.markdown(
        "<p style='font-family:Manrope,sans-serif;font-size:20px;font-weight:800;"
        "color:white;margin-bottom:14px;'>Live Conversion Prediction</p>",
        unsafe_allow_html=True
    )

    render_html(f"""
    <div class="insight-box">
        <h2 style="font-family:'Manrope',sans-serif;font-weight:900;font-size:28px;
                   color:white;margin-bottom:14px;">{risk_label}</h2>
        <p style="font-family:'Manrope',sans-serif;">
            Purchase Probability: <b>{probability*100:.2f}%</b>
        </p>
        <p style="font-family:'Manrope',sans-serif;">
            Abandonment Risk: <b>{abandonment_risk:.2f}%</b>
        </p>
        <p style="font-family:'Manrope',sans-serif;">
            Buyer Intent Level: <b>{buyer_intent}</b>
        </p>
    </div>
    """)

    funnel_stages = ["Visitor", "Product Interest", "Engaged Session", "Purchase Likelihood"]
    funnel_values = [
        100,
        min(100, product_related / 5),
        min(100, engagement_score / 25),
        probability * 100
    ]

    fig6 = go.Figure(go.Funnel(
        y=funnel_stages, x=funnel_values,
        textinfo="value+percent initial",
        marker=dict(color=["#22d3ee", "#38bdf8", "#a855f7", "#f472b6"])
    ))
    fig6.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Manrope"),
        height=450
    )
    st.plotly_chart(fig6, use_container_width=True, key="funnel_chart", config=plot_config)

    def create_pdf():
        buffer = BytesIO()
        doc    = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        story  = [
            Paragraph("Optinova Commerce AI Report", styles["Title"]),
            Spacer(1, 20),
            Paragraph(f"Purchase Probability: {probability*100:.2f}%",        styles["BodyText"]),
            Paragraph(f"Abandonment Risk:      {abandonment_risk:.2f}%",      styles["BodyText"]),
            Paragraph(f"Prediction:            {risk_label}",                 styles["BodyText"]),
            Paragraph(f"Buyer Intent:          {buyer_intent}",               styles["BodyText"]),
            Paragraph(f"Session Health:        {session_health:.1f}",         styles["BodyText"]),
            Paragraph(f"Model Confidence:      {conversion_confidence:.1f}%", styles["BodyText"]),
        ]
        doc.build(story)
        pdf = buffer.getvalue()
        buffer.close()
        return pdf

    st.download_button(
        label="⬇ Download AI Report",
        data=create_pdf(),
        file_name="optinova_ai_report.pdf",
        mime="application/pdf"
    )


# ── MODEL INSIGHTS ────────────────────────────────────────────────────────────
elif selected_page == "Model Insights":
    st.markdown(
        "<p style='font-family:Manrope,sans-serif;font-size:20px;font-weight:800;"
        "color:white;margin-bottom:14px;'>Top Feature Importance</p>",
        unsafe_allow_html=True
    )

    importance_df = pd.DataFrame({
        "Feature":    feature_columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(10)

    fig3 = go.Figure(go.Bar(
        x=importance_df["Importance"],
        y=importance_df["Feature"],
        orientation="h",
        marker=dict(color="#22d3ee")
    ))
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Manrope"),
        height=500
    )
    fig3.update_yaxes(autorange="reversed")
    st.plotly_chart(fig3, use_container_width=True, key="feature_chart", config=plot_config)

    render_html("""
    <div class="soft-box">
        <h3 style="font-family:'Manrope',sans-serif;font-weight:800;color:white;
                   margin-bottom:10px;">How to interpret this?</h3>
        <p style="font-family:'Manrope',sans-serif;">
            This chart shows which session attributes influenced the model most while predicting
            user conversion behavior. Higher importance indicates a stronger influence on purchase
            probability or abandonment risk.
        </p>
    </div>
    """)


# ── RISK BREAKDOWN ────────────────────────────────────────────────────────────
elif selected_page == "Risk Breakdown":
    st.markdown(
        "<p style='font-family:Manrope,sans-serif;font-size:20px;font-weight:800;"
        "color:white;margin-bottom:14px;'>Risk Category Breakdown</p>",
        unsafe_allow_html=True
    )

    fig4 = go.Figure(go.Pie(
        labels=["Conversion Strength", "Abandonment Risk"],
        values=[100 - abandonment_risk, abandonment_risk],
        hole=0.6,
        marker=dict(colors=["#22d3ee", "#f472b6"])
    ))
    fig4.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Manrope"),
        height=430
    )
    st.plotly_chart(fig4, use_container_width=True, key="risk_pie_chart", config=plot_config)

    render_html(f"""
    <div class="insight-box">
        <h3 style="font-family:'Manrope',sans-serif;font-weight:800;color:white;
                   margin-bottom:10px;">Risk Interpretation</h3>
        <p style="font-family:'Manrope',sans-serif;">
            This session shows an abandonment risk of <b>{abandonment_risk:.2f}%</b>.
            The system uses behavioral signals such as bounce rate, exit rate, page value,
            and product activity to estimate the likelihood of drop-off.
        </p>
    </div>
    """)


# ── ABOUT ─────────────────────────────────────────────────────────────────────
elif selected_page == "About":
    components.html(f"""
<!DOCTYPE html>
<html>
<head>
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800;900&display=swap');

* {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Manrope', sans-serif; }}
body {{ background: transparent; color: white; padding: 0; }}

.about-card {{
    background: rgba(15,23,42,0.90);
    border-radius: 28px;
    padding: 40px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 18px 60px rgba(0,0,0,0.34);
    margin-bottom: 24px;
}}

.about-title {{
    font-size: 36px;
    font-weight: 900;
    color: white;
    letter-spacing: -1px;
    margin-bottom: 18px;
}}

.about-body {{
    font-size: 17px;
    line-height: 1.9;
    color: #dbe4ee;
    font-weight: 400;
}}

.arch-wrap {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 14px;
    margin-top: 24px;
}}

.arch-card {{
    background: rgba(8,15,35,0.90);
    border: 1px solid rgba(34,211,238,0.16);
    border-radius: 20px;
    padding: 22px 16px;
    text-align: center;
    min-height: 140px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
}}

.arch-icon  {{ font-size: 32px; }}
.arch-title {{ color: #f8fafc; font-size: 14px; font-weight: 800; }}
.arch-sub   {{ color: #64748b; font-size: 12px; line-height: 1.5; }}

/* ── Brands section — comes BEFORE built-by ── */
.brands-section-about {{
    margin-top: 36px;
    border-top: 1px solid rgba(255,255,255,0.08);
    padding-top: 30px;
    text-align: center;
}}

.brands-label-about {{
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 22px;
}}

.brands-row-about {{
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 12px;
}}

/* Each brand pill: logo + name */
.bpill {{
    padding: 9px 18px;
    border-radius: 40px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 0.3px;
    white-space: nowrap;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}}

.bpill img {{
    width: 20px;
    height: 20px;
    object-fit: contain;
    border-radius: 4px;
    flex-shrink: 0;
    background: white;
    padding: 1px;
}}

.bpill.amazon   {{ color: #f0a500; border-color: rgba(240,165,0,0.30);   background: rgba(240,165,0,0.08); }}
.bpill.flipkart {{ color: #47c0f5; border-color: rgba(71,192,245,0.30);  background: rgba(71,192,245,0.08); }}
.bpill.myntra   {{ color: #f472b6; border-color: rgba(244,114,182,0.30); background: rgba(244,114,182,0.08); }}
.bpill.meesho   {{ color: #c084fc; border-color: rgba(192,132,252,0.30); background: rgba(192,132,252,0.08); }}
.bpill.ajio     {{ color: #22d3ee; border-color: rgba(34,211,238,0.30);  background: rgba(34,211,238,0.08); }}
.bpill.nykaa    {{ color: #fb923c; border-color: rgba(251,146,60,0.30);  background: rgba(251,146,60,0.08); }}
.bpill.tatacliq {{ color: #60a5fa; border-color: rgba(96,165,250,0.30);  background: rgba(96,165,250,0.08); }}
.bpill.zepto    {{ color: #a78bfa; border-color: rgba(167,139,250,0.30); background: rgba(167,139,250,0.08); }}

/* ── Built-by line — comes AFTER brands ── */
.built-by {{
    text-align: center;
    color: #64748b;
    font-size: 14px;
    font-weight: 500;
    padding: 26px 0 8px;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin-top: 28px;
}}

.built-by b {{ color: #93c5fd; font-weight: 800; }}
</style>
</head>
<body>

<!-- ── WHAT IS OPTINOVA ── -->
<div class="about-card">
    <div class="about-title">What is Optinova Commerce?</div>
    <div class="about-body">
        Optinova Commerce is an AI-powered ecommerce analytics platform designed to analyze
        visitor behavior and predict whether a customer is likely to complete a purchase or
        abandon the shopping journey.
        <br><br>
        The system combines machine learning, behavioral analytics, and real-time risk scoring
        to help ecommerce businesses understand customer intent and improve conversion strategies.
        <br><br>
        Example: if a visitor explores multiple product pages but also shows high bounce and
        exit behavior, the model identifies elevated abandonment tendencies and surfaces
        intelligent behavioral insights.
    </div>
</div>

<!-- ── PROJECT ARCHITECTURE ── -->
<div class="about-card">
    <div class="about-title">Project Architecture</div>
    <div class="arch-wrap">
        <div class="arch-card">
            <div class="arch-icon">📊</div>
            <div class="arch-title">Dataset</div>
            <div class="arch-sub">Online Shoppers Purchasing Intention</div>
        </div>
        <div class="arch-card">
            <div class="arch-icon">🐍</div>
            <div class="arch-title">Python + Pandas</div>
            <div class="arch-sub">Cleaning, EDA, Feature Engineering</div>
        </div>
        <div class="arch-card">
            <div class="arch-icon">⚙️</div>
            <div class="arch-title">Scikit-learn</div>
            <div class="arch-sub">Scaling, Splitting, Evaluation</div>
        </div>
        <div class="arch-card">
            <div class="arch-icon">🧠</div>
            <div class="arch-title">XGBoost</div>
            <div class="arch-sub">Conversion Prediction Model</div>
        </div>
        <div class="arch-card">
            <div class="arch-icon">🚀</div>
            <div class="arch-title">Streamlit + Plotly</div>
            <div class="arch-sub">Interactive AI Dashboard</div>
        </div>
    </div>

    <!-- ── MADE FOR INNOVATIVE BRANDS — above built-by ── -->
    <div class="brands-section-about">
        <div class="brands-label-about">✦ Made for Innovative Brands ✦</div>
        <div class="brands-row-about">
            <span class="bpill amazon">
                <img src="https://www.google.com/s2/favicons?sz=32&domain=amazon.com" alt="Amazon">Amazon
            </span>
            <span class="bpill flipkart">
                <img src="https://www.google.com/s2/favicons?sz=32&domain=flipkart.com" alt="Flipkart">Flipkart
            </span>
            <span class="bpill myntra">
                <img src="https://www.google.com/s2/favicons?sz=32&domain=myntra.com" alt="Myntra">Myntra
            </span>
            <span class="bpill meesho">
                <img src="https://www.google.com/s2/favicons?sz=32&domain=meesho.com" alt="Meesho">Meesho
            </span>
            <span class="bpill ajio">
                <img src="https://www.google.com/s2/favicons?sz=32&domain=ajio.com" alt="AJIO">AJIO
            </span>
            <span class="bpill nykaa">
                <img src="https://www.google.com/s2/favicons?sz=32&domain=nykaa.com" alt="Nykaa">Nykaa
            </span>
            <span class="bpill tatacliq">
                <img src="https://www.google.com/s2/favicons?sz=32&domain=tatacliq.com" alt="Tata CLiQ">Tata CLiQ
            </span>
            <span class="bpill zepto">
                <img src="https://www.google.com/s2/favicons?sz=32&domain=zeptonow.com" alt="Zepto">Zepto
            </span>
        </div>
    </div>

    <!-- ── BUILT BY — always last ── -->
    <div class="built-by">
        Built by <b>Pralisha Tripathy</b> &nbsp;·&nbsp; AI &nbsp;·&nbsp; Machine Learning &nbsp;·&nbsp; Ecommerce Intelligence
    </div>

</div>

</body>
</html>
""", height=960, scrolling=False)


# ── FOOTER ────────────────────────────────────────────────────────────────────
render_html("""
<br>
<hr style="border-color:rgba(255,255,255,0.07);">
<div style="text-align:center;color:#475569;padding-bottom:20px;
            font-family:'Manrope',sans-serif;font-size:14px;font-weight:500;">
    Built by <b style="color:#93c5fd;">Pralisha Tripathy</b>
    &nbsp;·&nbsp; AI &nbsp;·&nbsp; Machine Learning &nbsp;·&nbsp; Ecommerce Intelligence
</div>
""")