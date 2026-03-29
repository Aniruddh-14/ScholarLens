import streamlit as st
from src.summarizer import summarize
from src.utils import SAMPLE_TEXTS, word_count, char_count, extract_text_from_file
from src.visualization import generate_cluster_plot, get_top_keywords

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScholarLens",
    page_icon="📝",
    layout="wide",
)

# ── Theme state ──────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode


is_dark = st.session_state.dark_mode

# ── Theme colours ────────────────────────────────────────────────────
if is_dark:
    bg              = "#0c0e18"                          # deeper, true dark
    bg_secondary    = "#13162a"
    text_primary    = "#edf0fc"                          # brighter — better contrast
    text_secondary  = "#8892b0"                          # cooler slate, easier to read
    card_bg         = "rgba(18, 21, 44, 0.78)"           # deeper glass pane
    card_border     = "rgba(130, 150, 255, 0.13)"        # softer, less saturated
    card_value      = "#b8c8ff"                          # lighter indigo for contrast
    summary_bg      = "rgba(18, 21, 44, 0.65)"
    summary_border  = "#7a8ef5"                          # slightly lighter accent
    input_bg        = "#10132a"
    input_border    = "#222545"                          # muted, softer
    input_text      = "#edf0fc"
    divider_color   = "rgba(255,255,255,0.05)"
    sidebar_bg      = "#0a0c1a"                          # darkest element
    hover_glow      = "rgba(102, 126, 234, 0.22)"        # stronger glow
    bullet_bg       = "rgba(102,126,234,0.07)"
    insight_bg      = "rgba(118,75,162,0.09)"
    insight_border  = "rgba(130,100,220,0.28)"           # softer insight border
    pill_active_bg  = "linear-gradient(135deg,#5a74e8,#6e42a8)"  # slightly muted active
    pill_inactive   = "rgba(18,21,44,0.65)"
    pill_inactive_b = "rgba(110,130,240,0.18)"
else:
    bg              = "#f8f9fc"
    bg_secondary    = "#ffffff"
    text_primary    = "#1e1e2e"
    text_secondary  = "#64748b"
    card_bg         = "rgba(255, 255, 255, 0.7)"
    card_border     = "rgba(102, 126, 234, 0.15)"
    card_value      = "#4a4a8a"
    summary_bg      = "rgba(240, 244, 255, 0.8)"
    summary_border  = "#667eea"
    input_bg        = "#ffffff"
    input_border    = "#e2e8f0"
    input_text      = "#1e1e2e"
    divider_color   = "rgba(0,0,0,0.06)"
    sidebar_bg      = "#f1f3f8"
    hover_glow      = "rgba(102, 126, 234, 0.08)"
    bullet_bg       = "rgba(102,126,234,0.06)"
    insight_bg      = "rgba(118,75,162,0.06)"
    insight_border  = "rgba(118,75,162,0.20)"
    pill_active_bg  = "linear-gradient(135deg,#667eea,#764ba2)"
    pill_inactive   = "rgba(255,255,255,0.8)"
    pill_inactive_b = "rgba(102,126,234,0.15)"

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ─── Global ─── */
    html, body, .stApp {{
        font-family: 'Inter', sans-serif;
        background-color: {bg};
        color: {text_primary};
        transition: background-color 0.45s ease, color 0.45s ease;
    }}
    .stApp > header {{ background: transparent !important; }}

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        transition: background-color 0.45s ease;
    }}
    section[data-testid="stSidebar"] * {{
        color: {text_primary} !important;
        transition: color 0.45s ease;
    }}

    /* ─── Animated header ─── */
    .main-header {{ text-align:center; padding:2rem 0 1rem; }}
    .main-header h1 {{
        background: linear-gradient(135deg,#7a8ef5,#9b6ec9,#d17ff0,#7a8ef5);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem; font-weight: 800; letter-spacing: -0.02em;
        animation: gradientShift 8s ease infinite;  /* slower = softer feel */
    }}
    .main-header p {{
        color: {text_secondary}; font-size:1.1rem; font-weight:400;
        margin-top:0.3rem; letter-spacing:0.01em;
        animation: fadeSlideUp 0.8s ease both;
    }}
    @keyframes gradientShift {{
        0%   {{ background-position: 0% 50%; }}
        50%  {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    @keyframes fadeSlideUp {{
        from {{ opacity:0; transform:translateY(12px); }}
        to   {{ opacity:1; transform:translateY(0); }}
    }}
    @keyframes fadeIn {{
        from {{ opacity:0; transform:translateY(8px); }}
        to   {{ opacity:1; transform:translateY(0); }}
    }}

    /* ─── AI Loading Spinner ─── */
    @keyframes orbit {{
        0%   {{ transform: rotate(0deg)   translateX(22px) rotate(0deg);   }}
        100% {{ transform: rotate(360deg) translateX(22px) rotate(-360deg); }}
    }}
    @keyframes orbit2 {{
        0%   {{ transform: rotate(120deg)  translateX(22px) rotate(-120deg);  }}
        100% {{ transform: rotate(480deg)  translateX(22px) rotate(-480deg);  }}
    }}
    @keyframes orbit3 {{
        0%   {{ transform: rotate(240deg)  translateX(22px) rotate(-240deg);  }}
        100% {{ transform: rotate(600deg)  translateX(22px) rotate(-600deg);  }}
    }}
    @keyframes pulseCore {{
        0%,100% {{ transform:scale(1);   opacity:1;   }}
        50%      {{ transform:scale(1.3); opacity:0.7; }}
    }}
    @keyframes shimmer {{
        0%   {{ background-position: -200% center; }}
        100% {{ background-position:  200% center; }}
    }}
    .ai-spinner-wrap {{
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        padding: 2.5rem 0; gap: 1.2rem;
    }}
    .ai-spinner {{
        position: relative; width: 60px; height: 60px;
    }}
    .ai-spinner-core {{
        position: absolute; top:50%; left:50%;
        width:14px; height:14px; border-radius:50%;
        background: linear-gradient(135deg,#667eea,#764ba2);
        transform: translate(-50%,-50%);
        animation: pulseCore 1.4s ease-in-out infinite;
        box-shadow: 0 0 18px rgba(102,126,234,0.6);
    }}
    .ai-spinner-dot {{
        position:absolute; top:50%; left:50%;
        width:8px; height:8px; border-radius:50%; margin:-4px;
    }}
    .ai-spinner-dot:nth-child(2) {{
        background:#667eea;
        animation: orbit 1.4s linear infinite;
        box-shadow: 0 0 8px rgba(102,126,234,0.8);
    }}
    .ai-spinner-dot:nth-child(3) {{
        background:#a78bfa;
        animation: orbit2 1.4s linear infinite;
        box-shadow: 0 0 8px rgba(167,139,250,0.8);
    }}
    .ai-spinner-dot:nth-child(4) {{
        background:#f093fb;
        animation: orbit3 1.4s linear infinite;
        box-shadow: 0 0 8px rgba(240,147,251,0.8);
    }}
    .ai-spinner-label {{
        font-size:0.9rem; font-weight:600; letter-spacing:0.08em;
        background: linear-gradient(90deg,#667eea,#764ba2,#f093fb,#667eea);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 2s linear infinite;
    }}

    /* ─── Mode pill toggle ─── */
    .mode-pills {{
        display: flex; gap: 0.5rem; flex-wrap: wrap;
        margin-bottom: 1rem;
    }}
    .mode-pill {{
        padding: 0.45rem 1.1rem;
        border-radius: 50px;
        font-size: 0.88rem; font-weight: 600;
        cursor: pointer; border: 1.5px solid;
        transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
        user-select: none;
    }}
    .mode-pill.active {{
        background: {pill_active_bg};
        border-color: transparent; color: #fff;
        box-shadow: 0 4px 14px rgba(102,126,234,0.4);
        transform: translateY(-1px);
    }}
    .mode-pill.inactive {{
        background: {pill_inactive};
        border-color: {pill_inactive_b};
        color: {text_secondary};
    }}
    .mode-pill.inactive:hover {{
        border-color: #667eea; color: {text_primary};
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.15);
    }}

    /* ─── Metric cards (glassmorphism) ─── */
    .metric-card {{
        background: {card_bg};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {card_border};
        border-radius: 16px; padding: 1.4rem 1rem; text-align: center;
        transition: transform 0.25s cubic-bezier(0.4,0,0.2,1),
                    box-shadow 0.25s ease, background 0.45s ease;
        animation: fadeIn 0.5s ease both;
    }}
    .metric-card:hover {{
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 12px 35px {hover_glow};
    }}
    .metric-card h3 {{
        margin:0; font-size:2rem; font-weight:700;
        color:{card_value}; transition: color 0.45s ease;
    }}
    .metric-card p {{
        margin:0.3rem 0 0; color:{text_secondary};
        font-size:0.85rem; font-weight:500;
        text-transform:uppercase; letter-spacing:0.05em;
        transition: color 0.45s ease;
    }}

    /* ─── Summary box ─── */
    .summary-box {{
        background: {summary_bg};
        backdrop-filter: blur(10px);
        border-left: 4px solid {summary_border};
        border-radius: 12px; padding: 1.4rem 1.6rem;
        font-size: 1.05rem; line-height: 1.8; color: {text_primary};
        transition: background 0.45s ease, color 0.45s ease;
        animation: fadeIn 0.5s ease 0.1s both;
    }}

    /* ─── Bullet items ─── */
    .bullet-list {{ list-style:none; padding:0; margin:0; }}
    .bullet-item {{
        display: flex; align-items: flex-start; gap: 0.9rem;
        padding: 0.75rem 1rem; margin-bottom: 0.55rem;
        background: {bullet_bg};
        border-radius: 10px;
        border: 1px solid {card_border};
        font-size: 1rem; line-height: 1.6; color: {text_primary};
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        animation: fadeIn 0.4s ease both;
    }}
    .bullet-item:hover {{
        transform: translateX(4px);
        box-shadow: 0 4px 16px {hover_glow};
    }}
    .bullet-dot {{
        flex-shrink:0; width:8px; height:8px; border-radius:50%;
        background: linear-gradient(135deg,#667eea,#764ba2);
        margin-top: 0.45rem;
    }}

    /* ─── Insight cards ─── */
    .insight-card {{
        background: {insight_bg};
        border: 1px solid {insight_border};
        border-radius: 14px; padding: 1rem 1.3rem;
        margin-bottom: 0.75rem;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        animation: fadeIn 0.4s ease both;
    }}
    .insight-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(118,75,162,0.15);
    }}
    .insight-topic {{
        font-size: 0.78rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.1em;
        background: linear-gradient(135deg,#667eea,#764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.35rem;
    }}
    .insight-text {{
        font-size: 0.97rem; line-height: 1.65; color: {text_primary};
    }}

    /* ─── Buttons ─── */
    .stButton > button {{
        transition: transform 0.22s cubic-bezier(0.4,0,0.2,1),
                    box-shadow 0.22s ease, background 0.2s ease !important;
        position: relative; overflow: hidden;
    }}
    .stButton > button::after {{
        content:''; position:absolute; inset:0; border-radius:inherit;
        background: rgba(255,255,255,0.10);
        opacity:0; transition: opacity 0.2s ease;
    }}
    .stButton > button:hover::after {{ opacity:1; }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg,#667eea 0%,#764ba2 100%) !important;
        border: none !important; border-radius: 50px !important;
        padding: 0.6rem 2rem !important; font-weight:600 !important;
        font-family:'Inter',sans-serif !important; letter-spacing:0.02em !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        transform: translateY(-3px) scale(1.02) !important;
        /* multi-layer diffuse glow: tight core + soft outer bloom */
        box-shadow:
            0 0 0 1px rgba(102,126,234,0.25),
            0 4px 14px rgba(102,126,234,0.40),
            0 10px 40px rgba(118,75,162,0.25),
            0 20px 60px rgba(102,126,234,0.12) !important;
    }}
    .stButton > button[kind="primary"]:active {{
        transform: translateY(0) scale(0.97) !important;
        box-shadow: 0 2px 8px rgba(102,126,234,0.30) !important;
    }}
    .stButton > button:not([kind="primary"]):hover {{
        transform: translateY(-2px) !important;
        box-shadow:
            0 4px 14px {hover_glow},
            0 8px 28px rgba(102,126,234,0.08) !important;
    }}

    /* ─── Text area ─── */
    .stTextArea textarea {{
        background-color: {input_bg} !important;
        color: {input_text} !important;
        border-color: {input_border} !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        transition: background-color 0.45s ease, color 0.45s ease,
                    border-color 0.3s ease, box-shadow 0.3s ease !important;
    }}
    .stTextArea textarea:focus {{
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.18) !important;
    }}

    /* ─── Section headings ─── */
    .section-title {{
        font-size:1.3rem; font-weight:700; color:{text_primary};
        margin:1.5rem 0 1rem; display:flex; align-items:center; gap:0.5rem;
        animation: fadeSlideUp 0.5s ease both;
    }}

    /* ─── How-it-works steps ─── */
    .step-item {{ display:flex; align-items:flex-start; gap:0.75rem; padding:0.7rem 0; }}
    .step-num {{
        flex-shrink:0; width:28px; height:28px; border-radius:50%;
        background:linear-gradient(135deg,#667eea,#764ba2); color:white;
        display:flex; align-items:center; justify-content:center;
        font-size:0.8rem; font-weight:700;
    }}
    .step-text {{ font-size:0.9rem; color:{text_secondary}; line-height:1.5; }}
    .step-text strong {{ color:{text_primary}; }}

    /* ─── Divider ─── */
    hr {{ border-color: {divider_color} !important; }}

    /* ─── Footer ─── */
    .app-footer {{
        text-align:center; padding:2rem 0 1rem;
        color:{text_secondary}; font-size:0.82rem; letter-spacing:0.02em;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>📝 ScholarLens</h1>
        <p>Extractive summarization · TF-IDF &amp; K-Means clustering</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── Sidebar controls ────────────────────────────────────────────────
with st.sidebar:
    theme_icon  = "☀️" if is_dark else "🌙"
    theme_label = "Light Mode" if is_dark else "Dark Mode"
    st.button(f"{theme_icon}  {theme_label}", on_click=toggle_theme, use_container_width=True)
    st.divider()

    st.header("⚙️ Settings")
    summary_ratio = st.slider(
        "Summary ratio",
        min_value=0.1, max_value=0.8, value=0.3, step=0.05,
        help="Fraction of original sentences to keep in the summary.",
    )

    st.divider()
    st.header("📚 Sample Texts")
    sample_choice = st.selectbox("Load a sample", ["— Choose —"] + list(SAMPLE_TEXTS.keys()))

    st.divider()
    st.markdown(
        f"""
        <div style="margin-top:0.5rem;">
            <div class="step-item">
                <div class="step-num">1</div>
                <div class="step-text"><strong>Preprocess</strong> — clean &amp; split into sentences</div>
            </div>
            <div class="step-item">
                <div class="step-num">2</div>
                <div class="step-text"><strong>TF-IDF</strong> — score sentence importance</div>
            </div>
            <div class="step-item">
                <div class="step-num">3</div>
                <div class="step-text"><strong>K-Means</strong> — cluster sentences by topic</div>
            </div>
            <div class="step-item">
                <div class="step-num">4</div>
                <div class="step-text"><strong>Select</strong> — pick the best from each cluster</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Mode state ───────────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "extractive"

MODE_META = {
    "extractive":  {"label": "🔹 Extractive",  "desc": "Top sentences via TF-IDF + K-Means clustering"},
    "abstractive": {"label": "🤖 Abstractive", "desc": "Fused, flowing paragraph with transition bridges"},
    "bullet":      {"label": "📋 Bullet",      "desc": "Key sentences formatted as scannable bullet points"},
    "key_insights": {"label": "💡 Key Insights","desc": "Topic-labelled insight cards for quick scanning"},
}

# ── Mode pill toggle ─────────────────────────────────────────────────
st.markdown('<div class="section-title">🧠 Summarization Mode</div>', unsafe_allow_html=True)
pill_cols = st.columns(len(MODE_META))
for col, (mode_key, meta) in zip(pill_cols, MODE_META.items()):
    with col:
        if st.button(
            meta["label"],
            key=f"pill_{mode_key}",
            type="primary" if st.session_state.mode == mode_key else "secondary",
            use_container_width=True,
        ):
            st.session_state.mode = mode_key
            st.rerun()

active_mode = st.session_state.mode
st.caption(f"_{MODE_META[active_mode]['desc']}_")

st.divider()

# ── Main input area ──────────────────────────────────────────────────
st.markdown('<div class="section-title">📄 Document Input</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a document (.pdf or .txt) instead of pasting:", type=["pdf", "txt"])

if sample_choice and sample_choice != "— Choose —":
    default_text = SAMPLE_TEXTS[sample_choice]
else:
    default_text = ""

input_text = st.text_area(
    "Or paste your text below:",
    value=default_text,
    height=200,
    placeholder="Enter or paste a paragraph / article here…",
)

if uploaded_file is not None:
    try:
        input_text = extract_text_from_file(uploaded_file, uploaded_file.name)
        st.success(f"Successfully loaded '{uploaded_file.name}'!")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# ── Summarize button ─────────────────────────────────────────────────
col_btn, _ = st.columns([1, 4])
with col_btn:
    run = st.button("🚀 Summarize", type="primary", use_container_width=True)

if run:
    if not input_text or not input_text.strip():
        st.warning("Please enter some text to summarize.")
    else:
        # ── AI-style spinner ──────────────────────────────────────
        spinner_ph = st.empty()
        spinner_ph.markdown(
            """
            <div class="ai-spinner-wrap">
                <div class="ai-spinner">
                    <div class="ai-spinner-core"></div>
                    <div class="ai-spinner-dot"></div>
                    <div class="ai-spinner-dot"></div>
                    <div class="ai-spinner-dot"></div>
                </div>
                <div class="ai-spinner-label">ANALYZING · SUMMARIZING</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        result = summarize(input_text, ratio=summary_ratio, mode=active_mode)
        spinner_ph.empty()

        # ── Metrics row ───────────────────────────────────────────
        st.markdown('<div class="section-title">📊 Statistics</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="metric-card"><h3>{result["original_sentence_count"]}</h3>'
                f"<p>Original Sentences</p></div>",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-card"><h3>{result["summary_sentence_count"]}</h3>'
                f"<p>Summary Sentences</p></div>",
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-card"><h3>{int(result["compression_ratio"] * 100)}%</h3>'
                f"<p>Compression</p></div>",
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="metric-card"><h3>{word_count(result["summary"])}</h3>'
                f"<p>Summary Words</p></div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Tabs layout ───────────────────────────────────────────
        tab_summary, tab_analytics, tab_original = st.tabs(["📝 Summary", "📊 Analytics", "📄 Original Document"])
        
        with tab_summary:
            st.markdown(
                f'<div class="section-title">{MODE_META[active_mode]["label"]} &nbsp;Result</div>',
                unsafe_allow_html=True,
            )

            if active_mode == "bullet" and result["bullets"]:
                items_html = "".join(
                    f'<li class="bullet-item" style="animation-delay:{i*0.06}s">'
                    f'<span class="bullet-dot"></span><span>{b}</span></li>'
                    for i, b in enumerate(result["bullets"])
                )
                st.markdown(f'<ul class="bullet-list">{items_html}</ul>', unsafe_allow_html=True)

            elif active_mode == "key_insights" and result["insights"]:
                cards_html = "".join(
                    f'<div class="insight-card" style="animation-delay:{i*0.07}s">'
                    f'<div class="insight-topic">{ins["topic"]}</div>'
                    f'<div class="insight-text">{ins["insight"]}</div></div>'
                    for i, ins in enumerate(result["insights"])
                )
                st.markdown(cards_html, unsafe_allow_html=True)

            else:
                st.markdown(f'<div class="summary-box">{result["summary"]}</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button("📥 Download Summary", data=result["summary"], file_name="summary.txt", mime="text/plain")

        with tab_analytics:
            st.markdown('<div class="section-title">🔑 Top Keywords</div>', unsafe_allow_html=True)
            keywords = get_top_keywords(result.get("sentences", []), n_keywords=10)
            if keywords:
                kw_html = " ".join([f'<span style="background:rgba(102,126,234,0.15); border:1px solid #667eea; border-radius:12px; padding:4px 10px; margin:4px; display:inline-block; font-size:0.85rem; font-weight:600; color:{text_primary};">{k.title()}</span>' for k in keywords])
                st.markdown(kw_html, unsafe_allow_html=True)
            else:
                st.info("Not enough text to extract keywords.")
                
            st.markdown('<div class="section-title">🕸️ Clustering Visualization</div>', unsafe_allow_html=True)
            if "sentences" in result and len(result["sentences"]) >= 3:
                fig = generate_cluster_plot(result["sentences"], result["n_keep"])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 3 sentences to generate clustering plot.")

        with tab_original:
            st.markdown(f"**Original Text** — {word_count(input_text)} words, {char_count(input_text)} chars")
            st.info(input_text)

    # Footer
    st.markdown(
        '<div class="app-footer">Built with Streamlit · TF-IDF · K-Means · ScholarLens</div>',
        unsafe_allow_html=True,
    )
