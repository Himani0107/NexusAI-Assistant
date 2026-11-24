import os
import time
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

from core.nlp_router import LocalNLPRouter
from core.rag_engine import DocumentRAGEngine
from core.llm_client import GeminiClient
from core.web_fallback import search_wikipedia

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("⚠️  WARNING: GEMINI_API_KEY not found in .env! Tier 2 will fall back to Tier 3.")
else:
    print("✅ GEMINI_API_KEY loaded successfully.")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-prod")

nlp_router = LocalNLPRouter(threshold=0.62)
rag_engine = DocumentRAGEngine()
gemini_client = GeminiClient(model_name="gemini-3.5-flash")


@app.route("/")
def index():
    session.setdefault("history", [])
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    num_chunks = rag_engine.ingest_file(file.stream, file.filename)
    if num_chunks == 0:
        return jsonify({"error": "Could not extract readable text from file."}), 400

    return jsonify({
        "message": f"Indexed '{file.filename}' into {num_chunks} searchable chunks.",
        "filename": file.filename,
        "chunks": num_chunks,
    })


@app.route("/chat", methods=["POST"])
def chat():
    start_time = time.perf_counter()
    data = request.get_json() or {}
    user_query = data.get("message", "").strip()

    if not user_query:
        return jsonify({"error": "Empty message"}), 400

    history = session.get("history", [])

    # 1. Tier 1: Local NLP Router (handles greetings, time, identity in ~5ms)
    nlp_result = nlp_router.predict(user_query)
    if nlp_result["matched"]:
        reply = nlp_result["reply"]
        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        _update_history(history, user_query, reply)
        return jsonify({
            "reply": reply,
            "tier": f"Tier 1: Local NLP ({nlp_result['tag']})",
            "confidence": nlp_result["confidence"],
            "latency_ms": latency_ms,
        })

    # 2. Retrieve RAG chunks only if query actually matches or references the uploaded doc
    rag_chunks = rag_engine.retrieve(user_query)

    # 3. Tier 2: Gemini 2.5 Flash (with RAG context + session memory)
    try:
        reply = gemini_client.generate(user_query, history, rag_chunks=rag_chunks)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        tier_label = (
         f"Tier 2: Gemini + RAG ({len(rag_chunks)} chunks)"
         if rag_chunks
         else "Tier 2: Gemini 3.5 Flash"
     )
        _update_history(history, user_query, reply)
        return jsonify({
            "reply": reply,
            "tier": tier_label,
            "confidence": rag_chunks[0]["score"] if rag_chunks else None,
            "latency_ms": latency_ms,
        })
    except Exception as e:
        print(f"[Tier 2 Gemini Error]: {e}")

    # 3.5 Offline RAG Rescue (if Gemini is down, still return the matched PDF excerpt!)
    if rag_chunks:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        best = rag_chunks[0]
        _update_history(history, user_query, best["text"])
        return jsonify({
            "reply": f"**From Document [Page {best['page']}]:**\n{best['text']}",
            "tier": "Tier 3: Offline RAG Excerpt",
            "confidence": best["score"],
            "latency_ms": latency_ms,
        })

    # 4. Tier 3: Zero-Key Wikipedia Fallback
    wiki_reply = search_wikipedia(user_query)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
    if wiki_reply:
        _update_history(history, user_query, wiki_reply)
        return jsonify({
            "reply": wiki_reply,
            "tier": "Tier 3: Web Fallback (Wikipedia)",
            "confidence": None,
            "latency_ms": latency_ms,
        })

    fallback_msg = "All tiers unavailable right now. Please check your GEMINI_API_KEY configuration."
    return jsonify({
        "reply": fallback_msg,
        "tier": "Fallback Error",
        "confidence": None,
        "latency_ms": latency_ms,
    })


def _update_history(history, user_msg, bot_msg):
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": bot_msg})
    session["history"] = history[-10:]


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)