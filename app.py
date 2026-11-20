
from pathlib import Path
import json
import os
import urllib.request

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from ollama import Client
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Airline Customer Review Intelligence", page_icon="✈️", layout="wide")

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(os.getenv("DATA_PATH", BASE_DIR / "data" / "skytrax_airline_review.csv"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", BASE_DIR / "rag_artifacts"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.20"))

SERVICE_AREAS = [
    "Flight disruption", "Refunds", "Ground service", "Baggage handling",
    "Cabin service", "Booking and ticketing", "Customer communication",
    "Accessibility assistance", "Safety or medical support", "Other"
]

class CustomerSupportResponse(BaseModel):
    customer_issue_summary: str
    conversation_history_summary: str
    service_area: str
    human_attention_required: str
    escalation_reason: str
    suggested_customer_response: str
    internal_recommended_action: str

@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    df.columns = df.columns.str.strip()
    if "review_text" not in df.columns:
        raise ValueError("Dataset must contain a 'review_text' column.")
    return df

@st.cache_resource(show_spinner=False)
def load_embedding_model(name: str):
    return SentenceTransformer(name)

@st.cache_data(show_spinner=False)
def load_rag_metadata(path: str) -> pd.DataFrame:
    return pd.read_parquet(Path(path) / "knowledge_base_metadata.parquet")

@st.cache_data(show_spinner=False)
def load_rag_embeddings(path: str) -> np.ndarray:
    return np.load(Path(path) / "knowledge_base_embeddings.npy").astype("float32")

@st.cache_data(show_spinner=False)
def load_manifest(path: str) -> dict:
    manifest_path = Path(path) / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}

@st.cache_resource(show_spinner=False)
def get_ollama_client(host: str):
    return Client(host=host)

def ollama_available(host: str) -> bool:
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=3)
        return True
    except Exception:
        return False

def detect_column(df, candidates):
    lookup = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None

def retrieve_reviews(query, model, metadata, embeddings, top_k=5, min_similarity=0.20):
    query_vector = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores = cosine_similarity(query_vector, embeddings)[0]
    result = metadata.copy()
    result["similarity_score"] = scores
    result = result.sort_values("similarity_score", ascending=False)
    result = result[result["similarity_score"] >= min_similarity]
    return result.head(top_k).reset_index(drop=True)

SYSTEM_PROMPT = """
You are an airline customer-support assistant. Use only the current review,
conversation history, and retrieved historical cases. Historical cases are reference
evidence and are not facts about the current passenger. Never copy personal details,
flight numbers, booking references, or claims from another case. Do not invent refund
status, compensation eligibility, investigation outcomes, or prior contacts. Show
empathy without admitting unsupported legal liability. Require human attention for
unresolved refunds, repeated failures, urgent disruptions, missed connections,
cancellation without confirmed rebooking, lost or significantly delayed baggage,
safety, security, medical, legal, accessibility, or material financial-loss issues.
Return only JSON matching the supplied schema.
"""

def generate_response(review_text, history, evidence, client):
    contexts = []
    for i, row in evidence.iterrows():
        case_id = row.get("case_id", f"CASE-{i+1}")
        document = str(row.get("document", row.get("review_text", "")))[:1500]
        contexts.append(f"Case {i+1} ({case_id})\n{document}\nSimilarity: {row['similarity_score']:.4f}")
    context = "\n\n".join(contexts) or "No sufficiently similar historical case was retrieved."
    prompt = f"""
CURRENT CUSTOMER REVIEW:
<current_review>{review_text}</current_review>

CONVERSATION HISTORY:
<history>{history}</history>

RETRIEVED CASES:
<cases>{context}</cases>

Return: a two-sentence issue summary, history summary, main service area, human-attention
Yes/No decision, escalation reason, a professional customer response with a practical
next step, and an internal recommended action.
"""
    response = client.chat(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        format=CustomerSupportResponse.model_json_schema(),
        options={"temperature": 0, "num_ctx": 8192},
    )
    return CustomerSupportResponse.model_validate_json(response.message.content).model_dump()

# -----------------------------
# Load application assets
# -----------------------------
st.title("✈️ Airline Customer Review Intelligence")
st.caption("Review analytics, semantic retrieval, and grounded customer-response generation")

try:
    df = load_dataset(str(DATA_PATH))
except Exception as exc:
    st.error(str(exc))
    st.info("Place skytrax_airline_review.csv in the data folder or set the DATA_PATH environment variable.")
    st.stop()

try:
    metadata = load_rag_metadata(str(ARTIFACT_DIR))
    embeddings = load_rag_embeddings(str(ARTIFACT_DIR))
    manifest = load_manifest(str(ARTIFACT_DIR))
    if len(metadata) != len(embeddings):
        raise ValueError("RAG metadata and embedding row counts do not match.")
except Exception as exc:
    st.error(f"RAG artifacts could not be loaded: {exc}")
    st.info("Copy knowledge_base_metadata.parquet, knowledge_base_embeddings.npy, and manifest.json into rag_artifacts.")
    st.stop()

embedding_name = manifest.get("embedding_model", EMBEDDING_MODEL_NAME)
embedding_model = load_embedding_model(embedding_name)
client = get_ollama_client(OLLAMA_HOST)

with st.sidebar:
    st.header("Filters and settings")
    airline_col = detect_column(df, ["airline", "airline_name"])
    rating_col = detect_column(df, ["overall", "rating", "overall_rating"])
    recommended_col = detect_column(df, ["recommended"])
    category_col = detect_column(df, ["nmf_topic_name", "category", "review_category", "primary_topic"])
    date_col = detect_column(df, ["date_flown", "date", "review_date", "published_date"])

    filtered = df.copy()
    if airline_col:
        airlines = sorted(filtered[airline_col].dropna().astype(str).unique())
        selected = st.multiselect("Airline", airlines)
        if selected:
            filtered = filtered[filtered[airline_col].astype(str).isin(selected)]
    if recommended_col:
        choices = sorted(filtered[recommended_col].dropna().astype(str).unique())
        rec = st.multiselect("Recommended", choices)
        if rec:
            filtered = filtered[filtered[recommended_col].astype(str).isin(rec)]
    top_k = st.slider("Retrieved cases", 1, 10, DEFAULT_TOP_K)
    threshold = st.slider("Minimum similarity", 0.0, 1.0, MIN_SIMILARITY, 0.05)
    st.divider()
    status = "Connected" if ollama_available(OLLAMA_HOST) else "Unavailable"
    st.write(f"**Ollama:** {status}")
    st.write(f"**Model:** {MODEL_NAME}")
    st.write(f"**Knowledge base:** {len(metadata):,} cases")

analytics_tab, assistant_tab, evidence_tab = st.tabs([
    "📊 Review dashboard", "🤖 RAG support assistant", "🔎 Knowledge-base explorer"
])

with analytics_tab:
    total = len(filtered)
    avg_rating = pd.to_numeric(filtered[rating_col], errors="coerce").mean() if rating_col else np.nan
    if recommended_col:
        rec_series = filtered[recommended_col].astype(str).str.lower().str.strip()
        recommendation_rate = rec_series.isin(["yes", "true", "1"]).mean() * 100
    else:
        recommendation_rate = np.nan
    complaint_count = int((filtered[recommended_col].astype(str).str.lower().str.strip() == "no").sum()) if recommended_col else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", f"{total:,}")
    c2.metric("Average rating", "N/A" if np.isnan(avg_rating) else f"{avg_rating:.1f}")
    c3.metric("Recommendation rate", "N/A" if np.isnan(recommendation_rate) else f"{recommendation_rate:.1f}%")
    c4.metric("Not recommended", f"{complaint_count:,}" if recommended_col else "N/A")

    left, right = st.columns(2)
    with left:
        st.subheader("Review categories")
        if category_col:
            counts = filtered[category_col].fillna("Unclassified").value_counts().reset_index()
            counts.columns = ["Category", "Reviews"]
            st.plotly_chart(px.bar(counts, x="Reviews", y="Category", orientation="h"), use_container_width=True)
        elif recommended_col:
            counts = filtered[recommended_col].fillna("Unknown").value_counts().reset_index()
            counts.columns = ["Recommendation", "Reviews"]
            st.plotly_chart(px.pie(counts, names="Recommendation", values="Reviews", hole=0.45), use_container_width=True)
        else:
            st.info("Add nmf_topic_name/category or recommended to display categories.")
    with right:
        st.subheader("Rating distribution")
        if rating_col:
            ratings = pd.to_numeric(filtered[rating_col], errors="coerce").dropna()
            st.plotly_chart(px.histogram(x=ratings, nbins=10, labels={"x": "Rating", "count": "Reviews"}), use_container_width=True)
        else:
            st.info("No rating column detected.")

    if date_col:
        st.subheader("Review trend")
        trend = filtered.copy()
        trend[date_col] = pd.to_datetime(trend[date_col], errors="coerce")
        trend = trend.dropna(subset=[date_col]).set_index(date_col).resample("ME").size().reset_index(name="Reviews")
        st.plotly_chart(px.line(trend, x=date_col, y="Reviews", markers=True), use_container_width=True)

    st.subheader("Review records")
    display_cols = [c for c in [airline_col, category_col, rating_col, recommended_col, date_col, "review_text"] if c]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

with assistant_tab:
    st.subheader("Understand a new review and generate a grounded response")
    with st.form("rag_form"):
        review = st.text_area("New customer review", height=180, placeholder="Paste the passenger review here...")
        history = st.text_area("Conversation history (optional)", value="No previous conversation history provided.", height=100)
        submitted = st.form_submit_button("Analyse and generate response", type="primary")

    if submitted:
        if not review.strip():
            st.warning("Enter a customer review.")
        elif not ollama_available(OLLAMA_HOST):
            st.error(f"Ollama is not reachable at {OLLAMA_HOST}. Start Ollama and pull {MODEL_NAME}.")
        else:
            with st.spinner("Retrieving relevant cases and generating a response..."):
                try:
                    evidence = retrieve_reviews(review, embedding_model, metadata, embeddings, top_k, threshold)
                    result = generate_response(review, history, evidence, client)
                    st.session_state["last_evidence"] = evidence
                    st.session_state["last_result"] = result
                except Exception as exc:
                    st.exception(exc)

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        a, b = st.columns([2, 1])
        with a:
            st.success(result["suggested_customer_response"])
        with b:
            st.metric("Service area", result["service_area"])
            st.metric("Human attention", result["human_attention_required"])
        st.markdown("**Customer issue summary**")
        st.write(result["customer_issue_summary"])
        st.markdown("**Escalation reason**")
        st.write(result["escalation_reason"])
        st.markdown("**Internal recommended action**")
        st.write(result["internal_recommended_action"])
        st.download_button("Download response as JSON", json.dumps(result, indent=2), "customer_support_response.json", "application/json")

with evidence_tab:
    st.subheader("Retrieved evidence")
    if "last_evidence" in st.session_state and not st.session_state["last_evidence"].empty:
        evidence = st.session_state["last_evidence"].copy()
        visible = [c for c in ["case_id", "nmf_topic_name", "similarity_score", "review_text", "document"] if c in evidence.columns]
        st.dataframe(evidence[visible], use_container_width=True, hide_index=True)
    else:
        st.info("Run an analysis to display the historical cases used by the RAG pipeline.")
