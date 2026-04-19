"""
Research Assistant page — ScholarLens Milestone 2.

A fully autonomous agentic AI research assistant powered by LangGraph,
Groq (LLaMA 3.1), and DuckDuckGo search.

Navigate to this page via the Streamlit sidebar navigation.
"""

import streamlit as st
from src.agent.graph import build_graph
from src.agent.state import ResearchState
from src.agent.export import to_markdown

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Assistant · ScholarLens",
    page_icon="🔬",
    layout="wide",
)

# ── CSS (shared visual style with main app) ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #0c0e18;
    color: #edf0fc;
}
.stApp > header { background: transparent !important; }
section[data-testid="stSidebar"] { background-color: #0a0c1a !important; }
section[data-testid="stSidebar"] * { color: #edf0fc !important; }

.ra-header { text-align: center; padding: 2rem 0 1rem; }
.ra-header h1 {
    background: linear-gradient(135deg,#7a8ef5,#9b6ec9,#d17ff0,#7a8ef5);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem; font-weight: 800; letter-spacing: -0.02em;
    animation: gradientShift 8s ease infinite;
}
.ra-header p { color: #8892b0; font-size: 1.05rem; margin-top: 0.3rem; }

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes fadeIn {
    from { opacity:0; transform:translateY(8px); }
    to   { opacity:1; transform:translateY(0); }
}

/* Report cards */
.report-section {
    background: rgba(18,21,44,0.78);
    border: 1px solid rgba(130,150,255,0.13);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.1rem;
    animation: fadeIn 0.5s ease both;
}
.report-section h3 {
    background: linear-gradient(135deg,#7a8ef5,#9b6ec9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 1.05rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; margin: 0 0 0.8rem;
}
.report-section p, .report-section li {
    color: #c8d0e8; font-size: 0.97rem; line-height: 1.8;
}
.report-section ul { padding-left: 1.2rem; margin: 0; }
.report-section li { margin-bottom: 0.4rem; }
.report-section a { color: #7a8ef5; text-decoration: none; }
.report-section a:hover { text-decoration: underline; }

/* Source pills */
.source-pill {
    display: inline-block;
    background: rgba(102,126,234,0.10);
    border: 1px solid rgba(102,126,234,0.25);
    border-radius: 8px;
    padding: 0.35rem 0.75rem;
    margin: 0.25rem;
    font-size: 0.82rem;
    color: #7a8ef5;
    word-break: break-all;
}

/* Follow-up chips */
.followup-chip {
    display: inline-block;
    background: rgba(118,75,162,0.15);
    border: 1px solid rgba(118,75,162,0.35);
    border-radius: 50px;
    padding: 0.4rem 1rem;
    margin: 0.3rem;
    font-size: 0.88rem;
    color: #c4a4f0;
    cursor: pointer;
    transition: all 0.2s ease;
}
.followup-chip:hover {
    background: rgba(118,75,162,0.30);
    transform: translateY(-2px);
}

/* Contradiction warning */
.contradiction-box {
    background: rgba(220,80,80,0.08);
    border: 1px solid rgba(220,80,80,0.25);
    border-radius: 12px;
    padding: 1rem 1.3rem;
    margin-bottom: 1rem;
}
.contradiction-box p { color: #f4a4a4; font-size: 0.92rem; margin: 0.25rem 0; }

/* Metric mini-cards */
.mini-card {
    background: rgba(18,21,44,0.78);
    border: 1px solid rgba(130,150,255,0.13);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.mini-card h4 { color: #b8c8ff; font-size: 1.6rem; font-weight: 700; margin: 0; }
.mini-card p  { color: #8892b0; font-size: 0.78rem; text-transform: uppercase;
                letter-spacing: 0.05em; margin: 0.2rem 0 0; }

/* Step log */
.step-log {
    background: rgba(10,12,26,0.9);
    border: 1px solid rgba(102,126,234,0.12);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #8892b0;
    line-height: 1.9;
}

hr { border-color: rgba(255,255,255,0.05) !important; }

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#667eea,#764ba2) !important;
    border: none !important; border-radius: 50px !important;
    padding: 0.6rem 2rem !important; font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(102,126,234,0.4) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="ra-header">
    <h1>🔬 AI Research Assistant</h1>
    <p>Agentic web research · LangGraph · Groq (LLaMA 3.1) · DuckDuckGo</p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Sidebar — API key + info ──────────────────────────────────────────
with st.sidebar:
    st.header("🔑 Setup")
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Free key at console.groq.com — takes ~30 seconds to create.",
    )
    st.caption("👉 [Get a free Groq key](https://console.groq.com)")
    st.divider()

    st.header("ℹ️ How It Works")
    steps_html = """
    <div style="margin-top:0.5rem;">
      <div style="display:flex;gap:0.75rem;padding:0.5rem 0;align-items:flex-start;">
        <div style="flex-shrink:0;width:26px;height:26px;border-radius:50%;
             background:linear-gradient(135deg,#667eea,#764ba2);color:white;
             display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;">1</div>
        <div style="font-size:0.85rem;color:#8892b0;line-height:1.5;">
          <strong style="color:#edf0fc;">Search</strong> — DuckDuckGo fetches top 5 sources</div>
      </div>
      <div style="display:flex;gap:0.75rem;padding:0.5rem 0;align-items:flex-start;">
        <div style="flex-shrink:0;width:26px;height:26px;border-radius:50%;
             background:linear-gradient(135deg,#667eea,#764ba2);color:white;
             display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;">2</div>
        <div style="font-size:0.85rem;color:#8892b0;line-height:1.5;">
          <strong style="color:#edf0fc;">Retrieve</strong> — Full page content is fetched</div>
      </div>
      <div style="display:flex;gap:0.75rem;padding:0.5rem 0;align-items:flex-start;">
        <div style="flex-shrink:0;width:26px;height:26px;border-radius:50%;
             background:linear-gradient(135deg,#667eea,#764ba2);color:white;
             display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;">3</div>
        <div style="font-size:0.85rem;color:#8892b0;line-height:1.5;">
          <strong style="color:#edf0fc;">Summarize</strong> — LLM condenses each source</div>
      </div>
      <div style="display:flex;gap:0.75rem;padding:0.5rem 0;align-items:flex-start;">
        <div style="flex-shrink:0;width:26px;height:26px;border-radius:50%;
             background:linear-gradient(135deg,#667eea,#764ba2);color:white;
             display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;">4</div>
        <div style="font-size:0.85rem;color:#8892b0;line-height:1.5;">
          <strong style="color:#edf0fc;">Validate</strong> — Facts cross-checked across sources</div>
      </div>
      <div style="display:flex;gap:0.75rem;padding:0.5rem 0;align-items:flex-start;">
        <div style="flex-shrink:0;width:26px;height:26px;border-radius:50%;
             background:linear-gradient(135deg,#667eea,#764ba2);color:white;
             display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;">5</div>
        <div style="font-size:0.85rem;color:#8892b0;line-height:1.5;">
          <strong style="color:#edf0fc;">Report</strong> — Structured research report generated</div>
      </div>
      <div style="display:flex;gap:0.75rem;padding:0.5rem 0;align-items:flex-start;">
        <div style="flex-shrink:0;width:26px;height:26px;border-radius:50%;
             background:linear-gradient(135deg,#667eea,#764ba2);color:white;
             display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;">6</div>
        <div style="font-size:0.85rem;color:#8892b0;line-height:1.5;">
          <strong style="color:#edf0fc;">Follow-up</strong> — Intelligent next questions suggested</div>
      </div>
    </div>
    """
    st.markdown(steps_html, unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────
if "ra_result" not in st.session_state:
    st.session_state.ra_result = None
if "ra_query" not in st.session_state:
    st.session_state.ra_query = ""

# ── Query input ───────────────────────────────────────────────────────
query = st.text_input(
    "🔎 Research Query",
    value=st.session_state.ra_query,
    placeholder="e.g. What are the latest advances in quantum computing?",
    label_visibility="collapsed",
)

col_btn, col_ex1, col_ex2, _ = st.columns([1.2, 1.4, 1.8, 3])
with col_btn:
    research_disabled = not groq_key or not query.strip()
    run = st.button(
        "🚀 Research",
        type="primary",
        use_container_width=True,
        disabled=research_disabled,
    )
with col_ex1:
    if st.button("💡 Example 1", use_container_width=True):
        st.session_state.ra_query = "What are the latest advances in quantum computing?"
        st.rerun()
with col_ex2:
    if st.button("🌍 Example 2", use_container_width=True):
        st.session_state.ra_query = "How does CRISPR gene editing work and what are its risks?"
        st.rerun()

if not groq_key:
    st.info("🔑 Enter your Groq API key in the sidebar to enable research. Get one free at [console.groq.com](https://console.groq.com).")

# ── Run the agent ─────────────────────────────────────────────────────
if run and query.strip() and groq_key:
    st.session_state.ra_query = query
    st.session_state.ra_result = None

    # Build initial state (include API key for nodes to access)
    initial_state: ResearchState = {
        "query": query.strip(),
        "search_results": [],
        "fetched_content": [],
        "source_summaries": [],
        "validated_facts": [],
        "contradictions": [],
        "report": {},
        "follow_up_questions": [],
        "error": None,
        "step_logs": [],
        "_api_key": groq_key,      # passed through state for node access
    }

    progress_ph = st.empty()
    graph = build_graph()

    # Stream node-by-node and update progress log live
    final_state = initial_state
    with st.status("🤖 Agent is researching…", expanded=True) as status:
        for chunk in graph.stream(initial_state):
            for node_name, node_output in chunk.items():
                final_state = {**final_state, **node_output}
                logs = final_state.get("step_logs", [])
                if logs:
                    st.write(logs[-1])

        if final_state.get("error"):
            status.update(label="❌ Research failed", state="error", expanded=True)
        else:
            status.update(label="✅ Research complete!", state="complete", expanded=False)

    st.session_state.ra_result = final_state

# ── Render results ────────────────────────────────────────────────────
result = st.session_state.ra_result

if result:
    if result.get("error"):
        st.error(f"**Research failed:** {result['error']}")
        st.caption("Try rephrasing your query or check your internet connection.")

    elif result.get("report"):
        report = result["report"]
        st.divider()

        # ── Metrics row ────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="mini-card"><h4>{len(result["search_results"])}</h4><p>Sources Found</p></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="mini-card"><h4>{len(result["source_summaries"])}</h4><p>Summarized</p></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="mini-card"><h4>{len(result["validated_facts"])}</h4><p>Validated Facts</p></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="mini-card"><h4>{len(report.get("findings", []))}</h4><p>Key Findings</p></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Report tabs ────────────────────────────────────────────
        tab_report, tab_sources, tab_logs = st.tabs(["📄 Report", "🔗 Sources", "🔧 Agent Log"])

        with tab_report:
            # Title
            st.markdown(f"## {report.get('title', 'Research Report')}")
            st.caption(f"Query: *{result['query']}*")
            st.divider()

            # Abstract
            if report.get("abstract"):
                st.markdown('<div class="report-section"><h3>📋 Abstract</h3>'
                            f'<p>{report["abstract"]}</p></div>', unsafe_allow_html=True)

            # Key Findings
            if report.get("findings"):
                findings_html = "".join(f"<li>{f}</li>" for f in report["findings"])
                st.markdown(f'<div class="report-section"><h3>🔑 Key Findings</h3>'
                            f'<ul>{findings_html}</ul></div>', unsafe_allow_html=True)

            # Contradictions
            if result.get("contradictions"):
                c_html = "".join(f'<p>⚠️ {c}</p>' for c in result["contradictions"])
                st.markdown(f'<div class="contradiction-box">'
                            f'<strong style="color:#f4a4a4;">Contradictions Detected</strong>'
                            f'{c_html}</div>', unsafe_allow_html=True)

            # Conclusion
            if report.get("conclusion"):
                st.markdown('<div class="report-section"><h3>🏁 Conclusion</h3>'
                            f'<p>{report["conclusion"]}</p></div>', unsafe_allow_html=True)

            st.divider()

            # Download button
            md_content = to_markdown(
                report,
                result["query"],
                result.get("follow_up_questions"),
                result.get("contradictions"),
            )
            safe_name = result["query"][:40].replace(" ", "_").replace("/", "-")
            st.download_button(
                "📥 Download as Markdown",
                data=md_content,
                file_name=f"research_{safe_name}.md",
                mime="text/markdown",
            )

            # Follow-up questions
            if result.get("follow_up_questions"):
                st.divider()
                st.markdown("### 💡 Follow-up Research Questions")
                st.caption("Click a question to use it as your next query.")
                chips_html = "".join(
                    f'<span class="followup-chip">{q}</span>'
                    for q in result["follow_up_questions"]
                )
                st.markdown(chips_html, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                # Clickable buttons for follow-up
                for q in result["follow_up_questions"]:
                    if st.button(f"↗ {q}", key=f"followup_{hash(q)}", use_container_width=True):
                        st.session_state.ra_query = q
                        st.session_state.ra_result = None
                        st.rerun()

        with tab_sources:
            st.markdown("### 🔗 Sources Consulted")
            for i, s in enumerate(result.get("source_summaries", []), 1):
                with st.expander(f"Source {i} — {s['url'][:70]}…"):
                    st.markdown(f"**URL:** [{s['url']}]({s['url']})")
                    st.markdown(f"**Summary:** {s['summary']}")

        with tab_logs:
            st.markdown("### 🔧 Agent Execution Log")
            logs = result.get("step_logs", [])
            log_html = "<br>".join(logs) if logs else "No logs available."
            st.markdown(f'<div class="step-log">{log_html}</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:#8892b0;font-size:0.82rem;">
    ScholarLens Research Assistant · LangGraph · Groq LLaMA 3.1 · DuckDuckGo
</div>
""", unsafe_allow_html=True)
