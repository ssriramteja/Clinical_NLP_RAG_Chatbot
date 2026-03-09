import streamlit as st
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure src/ is in the module path
sys.path.append(str(Path(__file__).parent / "src"))

from llm_chain import ClinicalRAGChain
from indexer import main as build_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialization ---
st.set_page_config(
    page_title="Clinical RAG Interface",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize RAG Chain (Cached to prevent re-loading on every interaction)
@st.cache_resource
def get_rag_chain():
    # If index doesn't exist, build it first
    index_path = Path("faiss_index")
    if not index_path.exists():
        logger.info("FAISS index not found. Building index...")
        build_index()
    
    return ClinicalRAGChain(k=4)

try:
    rag_chain = get_rag_chain()
except Exception as e:
    st.error(f"Failed to initialize RAG system: {e}")
    st.stop()

# --- CSS Styling ---
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
    .latency-tag {
        font-size: 0.8rem;
        color: #95a5a6;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "latencies" not in st.session_state:
    st.session_state.latencies = []

def fmt_latency(ms: float) -> str:
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms/1000:.1f}s"

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Clinical RAG System")
    st.caption("v1.1.0 | Standalone | Llama-3.3-70b")
    st.divider()

    st.markdown("#### System Status")
    st.success("RAG Engine Active")

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

# --- Execution Logic ---
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
                start_time = time.time()
                try:
                    if history:
                        result = rag_chain.ask_with_history(user_input, history)
                    else:
                        result = rag_chain.ask(user_input)
                    latency = round((time.time() - start_time) * 1000, 2)
                    
                    st.markdown(result["answer"])
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
                except Exception as e:
                    st.error(f"Inference error: {e}")

elif mode == "Risk Stratification":
    st.markdown("#### Patient Population Analytics")
    if st.button("Initialize Stratification Process", type="primary", use_container_width=True):
        with st.spinner("Running analytics..."):
            start_time = time.time()
            try:
                query = "Identify patients at elevated clinical risk. Categorize them into High, Medium, and Low risk cohorts."
                result = rag_chain.ask(query)
                st.success("Analysis complete.")
                st.markdown(result["answer"])
            except Exception as e:
                st.error(f"Risk analysis failed: {e}")

elif mode == "Chart Review":
    st.markdown("#### Individual Chart Retrieval")
    pid = st.text_input("Patient ID", placeholder="Example: P001").upper().strip()
    if st.button("Execute Lookup", type="primary"):
        if pid:
            with st.spinner(f"Retrieving data for {pid}..."):
                try:
                    if not pid.startswith("P") or not pid[1:].isdigit():
                         st.warning("Invalid format for Patient ID. Expected format: P[0-9]+")
                    else:
                        result = rag_chain.ask(f"Please provide a comprehensive clinical summary for patient {pid}.")
                        st.success("Record processing successful.")
                        st.markdown(result["answer"])
                except Exception as e:
                    st.error(f"Patient summarization failed: {e}")
        else:
            st.warning("Please enter a Patient ID.")
