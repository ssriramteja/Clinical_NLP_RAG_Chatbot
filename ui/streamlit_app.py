import streamlit as st
import requests
from datetime import datetime

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "ClinicalBot — RAG Chatbot",
    page_icon  = "🏥",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

API_URL = "http://localhost:8000"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 0.9rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .stat-box {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
if "messages"      not in st.session_state:
    st.session_state.messages      = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "latencies"     not in st.session_state:
    st.session_state.latencies     = []


# ── API Helpers ───────────────────────────────────────────────────────────────
def api_ask(question: str) -> dict:
    try:
        r = requests.post(
            f"{API_URL}/ask",
            json    = {"question": question, "verbose": False},
            timeout = 30
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "❌ Cannot connect to API. Is `uvicorn app.main:app` running?"}
    except requests.exceptions.Timeout:
        return {"error": "⏱️ Request timed out. Please try again."}
    except Exception as e:
        return {"error": str(e)}


def api_chat(question: str, history: list) -> dict:
    try:
        r = requests.post(
            f"{API_URL}/chat",
            json    = {"question": question, "chat_history": history},
            timeout = 30
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "❌ Cannot connect to API."}
    except Exception as e:
        return {"error": str(e)}


def api_risk() -> dict:
    try:
        r = requests.get(f"{API_URL}/risk", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_patient(pid: str) -> dict:
    try:
        r = requests.get(f"{API_URL}/patient/{pid}", timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError:
        return {"error": f"Patient {pid} not found or invalid ID."}
    except Exception as e:
        return {"error": str(e)}


def api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False


def fmt_latency(ms: float) -> str:
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms/1000:.1f}s"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 ClinicalBot")
    st.caption("Clinical NLP RAG Assistant")
    st.divider()

    # ── API Health ────────────────────────────────────────────────────────────
    st.markdown("### 🔌 API Status")
    if api_health():
        st.success("✅ FastAPI Connected")
    else:
        st.error("❌ API Offline")
        st.code("uvicorn app.main:app --reload", language="bash")

    st.divider()

    # ── Mode ──────────────────────────────────────────────────────────────────
    st.markdown("### 🎛️ Mode")
    mode = st.radio(
        "mode",
        options          = ["💬 Chat", "🚨 Risk Triage", "👤 Patient Lookup"],
        label_visibility = "collapsed"
    )

    st.divider()

    # ── Quick Questions (Chat mode only) ──────────────────────────────────────
    if mode == "💬 Chat":
        st.markdown("### ⚡ Quick Questions")
        quick_qs = [
            "Which patients need urgent care?",
            "What medications were prescribed?",
            "Which patient had a stroke?",
            "Summarize all HIGH risk patients",
            "What are P007's vitals?",
        ]
        for q in quick_qs:
            if st.button(q, use_container_width=True, key=f"q_{q[:15]}"):
                st.session_state["prefill"] = q

        st.divider()

    # ── Session Stats ─────────────────────────────────────────────────────────
    st.markdown("### 📊 Session Stats")
    avg = (sum(st.session_state.latencies) /
           len(st.session_state.latencies)
           if st.session_state.latencies else 0)

    col1, col2 = st.columns(2)
    col1.metric("Queries", st.session_state.total_queries)
    col2.metric("Avg Speed", fmt_latency(avg) if avg else "—")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.total_queries = 0
        st.session_state.latencies     = []
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🏥 ClinicalBot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    'Clinical NLP RAG · ClinicalBERT + FAISS + Groq Llama 3.3-70b'
    '</p>',
    unsafe_allow_html=True
)


# ══════════════════════════════════════════════════════
# MODE 1 — 💬 CHAT
# ══════════════════════════════════════════════════════
if mode == "💬 Chat":

    # Render existing messages
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "latency" in msg:
                st.caption(
                    f"⚡ {fmt_latency(msg['latency'])}  ·  "
                    f"📄 {msg.get('chunks', 4)} chunks  ·  "
                    f"🤖 {msg.get('model', 'llama-3.3-70b')}"
                )

    # Handle prefill from quick question button
    prefill    = st.session_state.pop("prefill", None)
    user_input = st.chat_input("Ask a clinical question...") or prefill

    if user_input:

        # Show user message immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Build history (exclude current message)
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        # Call API
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Retrieving clinical context..."):
                result = (
                    api_chat(user_input, history)
                    if history
                    else api_ask(user_input)
                )

            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown(result["answer"])

                latency = result.get("latency_ms", 0)
                chunks  = result.get("chunks_retrieved", 4)
                model   = result.get("model", "llama-3.3-70b")

                st.caption(
                    f"⚡ {fmt_latency(latency)}  ·  "
                    f"📄 {chunks} chunks  ·  "
                    f"🤖 {model}"
                )

                # Save to session
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": result["answer"],
                    "latency": latency,
                    "chunks":  chunks,
                    "model":   model,
                    "time":    datetime.now().strftime("%H:%M")
                })
                st.session_state.total_queries += 1
                st.session_state.latencies.append(latency)


# ══════════════════════════════════════════════════════
# MODE 2 — 🚨 RISK TRIAGE
# ══════════════════════════════════════════════════════
elif mode == "🚨 Risk Triage":

    st.markdown("## 🚨 Patient Risk Triage")
    st.markdown(
        "Runs a full risk analysis across all loaded clinical records "
        "and classifies patients by urgency."
    )

    if st.button("▶️ Run Risk Triage", type="primary", use_container_width=True):
        with st.spinner("🔍 Analyzing all patient records..."):
            result = api_risk()

        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"✅ Analysis complete — {fmt_latency(result['latency_ms'])}")
            st.divider()

            st.markdown("### 📋 Triage Report")
            st.markdown(result["answer"])

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Model",          "Llama 3.3-70b")
            c2.metric("Chunks Scanned", result.get("chunks_retrieved", "—"))
            c3.metric("Response Time",  fmt_latency(result["latency_ms"]))


# ══════════════════════════════════════════════════════
# MODE 3 — 👤 PATIENT LOOKUP
# ══════════════════════════════════════════════════════
elif mode == "👤 Patient Lookup":

    st.markdown("## 👤 Patient Record Lookup")
    st.markdown("Enter a Patient ID to retrieve a full structured clinical summary.")

    # Input row
    col1, col2 = st.columns([4, 1])
    with col1:
        pid = st.text_input(
            "pid",
            placeholder      = "Enter Patient ID — e.g. P001",
            label_visibility = "collapsed",
            max_chars        = 10
        ).upper().strip()
    with col2:
        lookup = st.button("🔍 Lookup", type="primary", use_container_width=True)

    # Quick select buttons
    st.markdown("**Quick Select:**")
    btn_cols = st.columns(5)
    all_pids = ["P001","P002","P003","P004","P005",
                "P006","P007","P008","P009","P010"]
    for i, p in enumerate(all_pids):
        with btn_cols[i % 5]:
            if st.button(p, key=f"pid_{p}", use_container_width=True):
                pid    = p
                lookup = True

    # Fetch and display
    if lookup and pid:
        with st.spinner(f"📂 Loading record for {pid}..."):
            result = api_patient(pid)

        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"✅ Record retrieved — {fmt_latency(result['latency_ms'])}")
            st.divider()
            st.markdown(f"### 📋 Clinical Summary — {pid}")
            st.markdown(result["answer"])

            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Response Time",  fmt_latency(result["latency_ms"]))
            c2.metric("Chunks Retrieved", result.get("chunks_retrieved", "—"))
