from pathlib import Path
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from ollama import Client
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SkySolve | Airline Customer Review Intelligence", page_icon="✈️", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", str(BASE_DIR / "rag_artefacts")))
DATA_URL = os.getenv("DATA_URL", "https://raw.githubusercontent.com/nyagayamonica/Skytrax_Airline_Review_Data/main/Skytrax%20Airline%20Review%20Data.zip")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_TOP_K = max(1, min(10, int(os.getenv("TOP_K", "5"))))
MIN_SIMILARITY = max(0.0, min(1.0, float(os.getenv("MIN_SIMILARITY", "0.20"))))

class CustomerSupportResponse(BaseModel):
    customer_issue_summary: str
    conversation_history_summary: str
    service_area: str
    human_attention_required: str
    escalation_reason: str
    suggested_customer_response: str
    internal_recommended_action: str

@st.cache_data(ttl=3600, show_spinner="Downloading Skytrax reviews...")
def load_dataset(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")]
        if not csv_names:
            raise FileNotFoundError("No CSV files found in the GitHub ZIP archive.")
        preferred = [name for name in csv_names if "skytrax" in Path(name).name.lower()]
        csv_name = preferred[0] if preferred else csv_names[0]
        with archive.open(csv_name) as stream:
            df = pd.read_csv(stream, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    if "review_text" not in df.columns:
        raise ValueError(f"CSV '{csv_name}' lacks review_text. Available columns: {list(df.columns)}")
    return df

@st.cache_resource(show_spinner="Loading semantic search model...")
def load_embedding_model(name: str):
    return SentenceTransformer(name)

@st.cache_data(show_spinner=False)
def load_rag_metadata(path: str) -> pd.DataFrame:
    return pd.read_parquet(Path(path) / "knowledge_base_metadata.parquet")

@st.cache_data(show_spinner=False)
def load_rag_embeddings(path: str) -> np.ndarray:
    return np.load(Path(path) / "knowledge_base_embeddings.npy", allow_pickle=False).astype("float32")

@st.cache_data(show_spinner=False)
def load_manifest(path: str) -> dict:
    manifest_path = Path(path) / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

@st.cache_resource(show_spinner=False)
def get_ollama_client(host: str):
    return Client(host=host, timeout=120)

def ollama_available(host: str) -> bool:
    try:
        client = get_ollama_client(host)
        client.list()
        return True
    except Exception:
        return False

def detect_column(df, candidates):
    lookup = {str(c).lower(): c for c in df.columns}
    return next((lookup[c.lower()] for c in candidates if c.lower() in lookup), None)

def retrieve_reviews(query, model, metadata, embeddings, top_k=5, min_similarity=0.20):
    vector = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    if embeddings.ndim != 2 or vector.shape[1] != embeddings.shape[1]:
        raise ValueError(f"Embedding dimension mismatch: query {vector.shape[1]}, knowledge base {embeddings.shape[1]}. Use the original embedding model.")
    scores = cosine_similarity(vector, embeddings)[0]
    result = metadata.copy()
    result["similarity_score"] = scores
    return result[result["similarity_score"] >= min_similarity].sort_values("similarity_score", ascending=False).head(top_k).reset_index(drop=True)

SYSTEM_PROMPT = """You are an airline customer-support assistant. Use only the current review,
conversation history, and retrieved historical cases. Historical cases are reference
evidence and are not facts about the current passenger. Never copy personal details,
flight numbers, booking references, or claims from another case. Do not invent refund
status, compensation eligibility, investigation outcomes, or prior contacts. Show
empathy without admitting unsupported legal liability. Require human attention for
unresolved refunds, repeated failures, urgent disruptions, missed connections,
cancellation without confirmed rebooking, lost or significantly delayed baggage,
safety, security, medical, legal, accessibility, or material financial-loss issues.
Return only JSON matching the supplied schema."""

def generate_response(review_text, history, evidence, client):
    contexts = []
    for i, row in evidence.iterrows():
        case_id = row.get("case_id", f"CASE-{i+1}")
        document = str(row.get("document", row.get("review_text", "")))[:1500]
        contexts.append(f"Case {i+1} ({case_id})\n{document}\nSimilarity: {row['similarity_score']:.4f}")
    context = "\n\n".join(contexts) or "No sufficiently similar historical case was retrieved."
    prompt = f"""CURRENT CUSTOMER REVIEW:\n<current_review>{review_text}</current_review>
CONVERSATION HISTORY:\n<history>{history}</history>
RETRIEVED CASES:\n<cases>{context}</cases>
Return: a two-sentence issue summary, history summary, main service area, human-attention
Yes/No decision, escalation reason, a professional customer response with a practical
next step, and an internal recommended action."""
    response = client.chat(model=MODEL_NAME,
                           messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                           format=CustomerSupportResponse.model_json_schema(),
                           options={"temperature": 0, "num_ctx": 8192})
    return CustomerSupportResponse.model_validate_json(response.message.content).model_dump()

st.title("✈️ SkySolve | Airline Customer Review Intelligence")
st.caption("Review analytics, semantic retrieval, and grounded customer-response generation")

try:
    df = load_dataset(DATA_URL)
except Exception as exc:
    st.error(f"Skytrax dataset could not be loaded: {exc}")
    st.info("Check that the GitHub raw ZIP URL is public and contains a CSV with a review_text column.")
    st.stop()

rag_error = None
metadata = None
embeddings = None
manifest = {}
try:
    metadata = load_rag_metadata(str(ARTIFACT_DIR))
    embeddings = load_rag_embeddings(str(ARTIFACT_DIR))
    manifest = load_manifest(str(ARTIFACT_DIR))
    if embeddings.ndim != 2 or len(metadata) != len(embeddings):
        raise ValueError("RAG metadata and embeddings must have matching row counts and a 2D embedding matrix.")
except Exception as exc:
    rag_error = str(exc)

embedding_name = manifest.get("embedding_model", EMBEDDING_MODEL_NAME)
if not isinstance(embedding_name, str) or not embedding_name:
    embedding_name = EMBEDDING_MODEL_NAME

with st.sidebar:
    st.header("Filters and settings")
    airline_col = detect_column(df, ["airline", "airline_name"])
    rating_col = detect_column(df, ["over_all_rating", "overall", "rating", "overall_rating"])
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
    if rag_error:
        st.warning("RAG artifacts unavailable")
    else:
        st.write(f"**Knowledge base:** {len(metadata):,} cases")
    st.write(f"**Ollama host:** {OLLAMA_HOST}")
    st.write(f"**Model:** {MODEL_NAME}")
    if OLLAMA_HOST.startswith("http://localhost") or OLLAMA_HOST.startswith("http://127.0.0.1"):
        st.caption("A localhost Ollama endpoint is inaccessible.")

analytics_tab, assistant_tab, evidence_tab = st.tabs(["📊 Review dashboard", "🤖 RAG support assistant", "🔎 Knowledge-base explorer"])

with analytics_tab:
    total = len(filtered)
    avg_rating = pd.to_numeric(filtered[rating_col], errors="coerce").mean() if rating_col else np.nan
    if recommended_col:
        rec_series = filtered[recommended_col].astype(str).str.lower().str.strip()
        recommendation_rate = rec_series.isin(["yes", "true", "1"]).mean() * 100 if total else np.nan
        complaint_count = int(rec_series.isin(["no", "false", "0"]).sum())
    else:
        recommendation_rate = np.nan
        complaint_count = 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", f"{total:,}")
    c2.metric("Average rating", "N/A" if pd.isna(avg_rating) else f"{avg_rating:.1f}")
    c3.metric("Recommendation rate", "N/A" if pd.isna(recommendation_rate) else f"{recommendation_rate:.1f}%")
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
            if len(ratings):
                st.plotly_chart(px.histogram(x=ratings, nbins=10, labels={"x": "Rating", "count": "Reviews"}), use_container_width=True)
            else:
                st.info("No numeric ratings in the current selection.")
        else:
            st.info("No rating column detected.")
    if date_col:
        st.subheader("Review trend")
        trend = filtered.copy()
        trend[date_col] = pd.to_datetime(trend[date_col], errors="coerce")
        trend = trend.dropna(subset=[date_col])
        if not trend.empty:
            trend = trend.set_index(date_col).resample("ME").size().reset_index(name="Reviews")
            st.plotly_chart(px.line(trend, x=date_col, y="Reviews", markers=True), use_container_width=True)
        else:
            st.info("No valid dates in the current selection.")
    st.subheader("Review records")
    display_cols = list(dict.fromkeys(c for c in [airline_col, category_col, rating_col, recommended_col, date_col, "review_text"] if c))
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

with assistant_tab:
    st.subheader("Understand a new review and generate a grounded response")
    if rag_error:
        st.error(f"RAG artifacts could not be loaded: {rag_error}")
        st.info(f"Check the files inside {ARTIFACT_DIR}. The dashboard remains available.")
    else:
        with st.form("rag_form"):
            review = st.text_area("New customer review", height=180, placeholder="Paste the passenger review here...")
            history = st.text_area("Conversation history (optional)", value="No previous conversation history provided.", height=100)
            submitted = st.form_submit_button("Analyse and generate response", type="primary")
        if submitted:
            if not review.strip():
                st.warning("Enter a customer review.")
            elif not ollama_available(OLLAMA_HOST):
                st.error(f"Ollama is not reachable at {OLLAMA_HOST}. Streamlit Cloud needs a remotely reachable inference server.")
            else:
                with st.spinner("Retrieving relevant cases and generating a response..."):
                    try:
                        embedding_model = load_embedding_model(embedding_name)
                        evidence = retrieve_reviews(review, embedding_model, metadata, embeddings, top_k, threshold)
                        result = generate_response(review, history, evidence, get_ollama_client(OLLAMA_HOST))
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
                st.write("**Service area:**", result["service_area"])
                st.write("**Human attention:**", result["human_attention_required"])
            st.markdown("**Customer issue summary**")
            st.write(result["customer_issue_summary"])
            st.markdown("**Conversation history summary**")
            st.write(result["conversation_history_summary"])
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
