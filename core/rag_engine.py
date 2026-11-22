import io
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class DocumentRAGEngine:
    def __init__(self):
        self.store = {}
        self.default_key = "__active_doc__"

    def ingest_file(self, file_stream, filename: str) -> int:
        file_bytes = file_stream.read()
        pages_text = []

        if filename.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(file_bytes))
            for idx, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages_text.append((idx, text))
        else:
            text = file_bytes.decode("utf-8", errors="ignore").strip()
            if text:
                pages_text.append((1, text))

        if not pages_text:
            return 0

        chunks = []
        for page_num, text in pages_text:
            words = text.split()
            chunk_size = 180
            overlap = 30
            for start in range(0, len(words), chunk_size - overlap):
                chunk_str = " ".join(words[start : start + chunk_size])
                if chunk_str.strip():
                    chunks.append({"page": page_num, "text": chunk_str})

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([c["text"] for c in chunks])

        self.store[self.default_key] = {
            "filename": filename,
            "chunks": chunks,
            "vectorizer": vectorizer,
            "matrix": matrix,
        }
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 3):
        doc = self.store.get(self.default_key)
        if not doc:
            return []

        q_vec = doc["vectorizer"].transform([query])
        scores = cosine_similarity(q_vec, doc["matrix"]).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.05:
                chunk = dict(doc["chunks"][idx])
                chunk["score"] = round(float(scores[idx]), 3)
                results.append(chunk)

        # Only fall back to opening chunks if user is explicitly asking about the uploaded doc
        doc_words = {
            "document", "pdf", "file", "summarize", "summary",
            "overview", "page", "uploaded", "text", "notes"
        }
        cleaned_words = query.lower().rstrip("?.!").split()
        if not results and any(w in doc_words for w in cleaned_words):
            for c in doc["chunks"][:top_k]:
                chunk = dict(c)
                chunk["score"] = 0.0
                results.append(chunk)

        return results