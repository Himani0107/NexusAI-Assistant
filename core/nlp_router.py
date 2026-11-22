import json
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LocalNLPRouter:
    def __init__(self, intents_path="data/intents.json", threshold=0.62):
        self.threshold = threshold
        self.patterns = []
        self.tags = []
        self.responses = {}

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base_dir, intents_path)

        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for intent in data.get("intents", []):
            tag = intent["tag"]
            self.responses[tag] = intent["responses"]
            for pattern in intent["patterns"]:
                self.patterns.append(pattern.lower())
                self.tags.append(tag)

        self.vectorizer = TfidfVectorizer()
        self.pattern_matrix = self.vectorizer.fit_transform(self.patterns)

    def predict(self, user_query: str):
        cleaned = user_query.lower().strip().rstrip("?.!")
        words = cleaned.split()

        # Bypass Tier 1 for longer queries (> 6 words) or document/RAG keywords
        rag_keywords = {
            "document", "pdf", "file", "upload", "uploaded",
            "summarize", "summary", "page", "explain"
        }
        if len(words) > 6 or any(w in rag_keywords for w in words):
            return {"matched": False, "confidence": 0.0}

        query_vec = self.vectorizer.transform([cleaned])
        similarities = cosine_similarity(query_vec, self.pattern_matrix).flatten()
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])

        if best_score >= self.threshold:
            tag = self.tags[best_idx]
            return {
                "matched": True,
                "tag": tag,
                "reply": self._resolve_reply(tag),
                "confidence": round(best_score, 3),
            }

        return {"matched": False, "confidence": round(best_score, 3)}

    def _resolve_reply(self, tag: str) -> str:
        now = datetime.now()
        if tag == "time_date":
            return f"The current time is {now.strftime('%I:%M %p')} on {now.strftime('%B %d, %Y')}."
        candidates = self.responses.get(tag, ["I'm here to help!"])
        reply = candidates[0]
        return (
            reply.replace("{time}", now.strftime("%I:%M %p"))
            .replace("{date}", now.strftime("%B %d, %Y"))
        )

    