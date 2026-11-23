# Skytrax Airline Review Intelligence

## Required files

Place the dataset at `data/skytrax_airline_review.csv` and these files in `rag_artefacts/`:

- `knowledge_base_metadata.parquet`
- `knowledge_base_embeddings.npy`
- `manifest.json`

The dataset must contain `review_text`. The dashboard automatically uses common fields such as `airline`, `overall`, `recommended`, `nmf_topic_name`, and `date_flown` when present.

## Local run

1. Install and start Ollama.
2. Run `ollama pull qwen2.5:1.5b`.
3. Create a Python 3.11 environment.
4. Run `pip install -r requirements.txt`.
5. Run `streamlit run app.py`.
6. Open `http://localhost:8501`.

## Docker deployment

1. Put the data and RAG artefacts in the indicated folders.
2. Run `docker compose up -d --build`.
3. Pull the model into the Ollama container:
   `docker compose exec ollama ollama pull qwen2.5:1.5b`
4. Open `http://localhost:8501`.

## Important deployment note

Streamlit Community Cloud can host the Streamlit interface, but a local `localhost` Ollama server on your computer is not reachable from that cloud runtime. For the current local Ollama design, use Docker on a persistent VM such as Azure VM, AWS EC2, or Google Compute Engine, or point `OLLAMA_HOST` to a secured remotely hosted Ollama service.

## Production controls

- Mask personal information before embedding or logging reviews.
- Put authentication and TLS in front of the app.
- Keep high-risk cases under human review.
- Restrict Ollama port 11434 to the application network.
- Evaluate retrieval relevance and escalation recall before customer-facing use.
- Version the dataset, artefacts, prompt, embedding model, and Ollama model.
