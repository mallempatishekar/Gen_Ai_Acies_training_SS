# ============================================================
# Script: build_vector_db.py
# Purpose: Build FAISS vector DB using local MiniLM embeddings (for Groq chatbot)
# Compatible with LangChain >= 0.2.15
# ============================================================

import os
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "chatbot_data" / "engagement_facts.csv"
VECTOR_DIR = BASE_DIR / "models" / "vector_store"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"❌ engagement_facts.csv not found at {DATA_PATH}")

# ============================================================
# Load CSV data and prepare documents
# ============================================================
df = pd.read_csv(DATA_PATH)
df.fillna("", inplace=True)

texts = []
for _, row in df.iterrows():
    text = (
        f"Post Title: {row.get('Post Title', '')}\n"
        f"Content Type: {row.get('Content Type', '')}\n"
        f"Engagement Rate: {row.get('Engagement Rate (%)', '')}%\n"
        f"Total Engagement: {row.get('Total Engagement', '')}\n"
        f"Impressions: {row.get('Impressions', '')}\n"
        f"Hashtags: {row.get('Hashtags', '')}\n"
        f"Created Date: {row.get('Created Date', '')}"
    )
    texts.append(text)

# Split text into manageable chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = [Document(page_content=t) for t in texts]
chunks = splitter.split_documents(docs)

# ============================================================
# Local Embeddings (no API required)
# ============================================================
print("🔹 Loading local embedding model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Custom wrapper to make it LangChain-compatible
class MiniLMEmbedder:
    def embed_documents(self, texts):
        return embedder.encode(texts, show_progress_bar=True).tolist()
    def embed_query(self, text):
        return embedder.encode([text])[0].tolist()

embeddings = MiniLMEmbedder()

# ============================================================
# Build FAISS vector database
# ============================================================
print("⚙️ Building FAISS index...")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save it locally
vectorstore.save_local(VECTOR_DIR)
print(f"✅ Vector database created successfully at: {VECTOR_DIR}")
