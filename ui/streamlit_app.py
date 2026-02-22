import streamlit as st
import requests
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Clinical RAG Interface",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

st.markdown("""
<style>
    .stApp {
        background-color: #f4f6f9;
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #7f8c8d;
        font-weight: 500;
        margin-bottom: 2rem;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e6ed;
    }
    .stat-box {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.0rem;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        background-color: #ffffff;
        border: 1px solid #eaebec;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .latency-tag {
        font-size: 0.8rem;
        color: #95a5a6;
    }
</style>
""", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "latencies" not in st.session_state:
    st.session_state.latencies = []


def api_ask(question: str) -> dict:
    try:
        response = requests.post(f"{API_URL}/ask", json={"question": question, "verbose": False}, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"API request failed: {e}")
        return {"error": "API error. Please ensure the backend is running."}

def api_chat(question: str, history: list) -> dict:
    try:
        response = requests.post(f"{API_URL}/chat", json={"question": question, "chat_history": history}, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"API request failed: {e}")
        return {"error": "API connection failed."}

def api_risk() -> dict:
    try:
        response = requests.get(f"{API_URL}/risk", timeout=45)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"API request failed: {e}")
        return {"error": "Risk analytics failed."}

def api_patient(pid: str) -> dict:
    try:
        response = requests.get(f"{API_URL}/patient/{pid}", timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as he:
        if response.status_code == 422:
            return {"error": f"Invalid format or ID not found: {pid}"}
        return {"error": f"HTTP Error: {he}"}
    except Exception as e:
        return {"error": str(e)}

def api_health() -> bool:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def fmt_latency(ms: float) -> str:
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms/1000:.1f}s"


with st.sidebar:
    st.markdown("### Clinical RAG System")
    st.caption("v1.0.0 | Llama-3.3-70b")
    st.divider()

    st.markdown("#### System Status")
    if api_health():
        st.success("API Connected")
    else:
        st.error("API Offline")

    st.divider()

    st.markdown("#### Operation Mode")
    mode = st.radio("mode", options=["Chat Interface", "Risk Stratification", "Chart Review"], label_visibility="collapsed")

    st.divider()

    st.markdown("#### Session Telemetry")
    avg_latency = sum(st.session_state.latencies) / len(st.session_state.latencies) if st.session_state.latencies else 0

    col1, col2 = st.columns(2)
    col1.metric("Queries", st.session_state.total_queries)
    col2.metric("Avg Latency", fmt_latency(avg_latency) if avg_latency else "N/A")

    if st.button("Clear Session Data", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.latencies = []
        st.rerun()

st.markdown('<p class="main-header">Clinical Inference Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Retrieval-Augmented Generation utilizing ClinicalBERT and FAISS</p>', unsafe_allow_html=True)


if mode == "Chat Interface":
    preset_queries = [
        "Are there any acute interventions required?",
        "Detail the medication regimen for diabetic patients.",
        "Synthesize high-risk patient indicators."
    ]
    st.markdown("**Suggested Queries:**")
    cols = st.columns(len(preset_queries))
    for idx, sample in enumerate(preset_queries):
        if cols[idx].button(sample, use_container_width=True):
            st.session_state["prefill"] = sample

    st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "latency" in msg:
                st.markdown(f'<p class="latency-tag">Process time: {fmt_latency(msg["latency"])} | Model: {msg["model"]}</p>', unsafe_allow_html=True)

    prefill = st.session_state.pop("prefill", None)
    user_input = st.chat_input("Input clinical query...") or prefill

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]

        with st.chat_message("assistant"):
            with st.spinner("Executing query pipeline..."):
                result = api_chat(user_input, history) if history else api_ask(user_input)

            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown(result["answer"])
                latency = result.get("latency_ms", 0)
                model = result.get("model", "llama-3.3-70b")
                st.markdown(f'<p class="latency-tag">Process time: {fmt_latency(latency)} | Model: {model}</p>', unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "latency": latency,
                    "model": model
                })
                st.session_state.total_queries += 1
                st.session_state.latencies.append(latency)


elif mode == "Risk Stratification":
    st.markdown("#### Patient Population Analytics")
    st.write("Executes cross-sectional evaluation of loaded clinical records to identify acuity metrics.")

    if st.button("Initialize Stratification Process", type="primary", use_container_width=True):
        with st.spinner("Running analytics..."):
            result = api_risk()

        if "error" in result:
            st.error(result["error"])
        else:
            st.success("Analysis complete.")
            st.markdown(result["answer"])


elif mode == "Chart Review":
    st.markdown("#### Individual Chart Retrieval")
    st.write("Input patient identifier to generate structured clinical summary.")

    col1, col2 = st.columns([4, 1])
    with col1:
        pid = st.text_input("pid", placeholder="Identifier format: P001", label_visibility="collapsed", max_chars=10).upper().strip()
    with col2:
        lookup = st.button("Execute", type="primary", use_container_width=True)

    if lookup and pid:
        with st.spinner(f"Retrieving data for {pid}..."):
            result = api_patient(pid)

        if "error" in result:
            st.error(result["error"])
        else:
            st.success("Record processing successful.")
            st.markdown(f"#### Synthesis output: {pid}")
            st.markdown(result["answer"])
