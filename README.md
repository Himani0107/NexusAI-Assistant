NEXUSAI ASSISTANT
Multi-Tier Conversational AI and Document Retrieval-Augmented Generation (RAG) System

Live Demo: https://nexusai-assistant.onrender.com

OVERVIEW
================================================================================

NexusAI Assistant is a full-stack conversational artificial intelligence and
document question-answering platform engineered with Python, Flask,
Scikit-Learn, and the Google Gemini API (google-genai).

Conventional AI assistants forward every user message directly to a large
language model, creating unnecessary network latency, high API token costs,
and frequent hallucinations when answering questions about private documents.
NexusAI Assistant solves these limitations through a deterministic, three-tier
query routing architecture that evaluates intent complexity, document relevance,
and service availability in real time.


SYSTEM ARCHITECTURE
================================================================================

    User Query
        |
        v
    [ Tier 1: Local NLP Intent Router (TF-IDF + Cosine Similarity) ]
        |
        +-- Matched (>= 0.62) -------> Instant Local Reply (~5-10 ms)
        |
        +-- Unmatched / Document Query
                |
                v
            [ Document RAG Retrieval (PyPDF + Overlapping TF-IDF Chunks) ]
                |
                v
            [ Tier 2: Gemini Model Cascade ]
            [ 3.5-Flash -> 3.1-Lite -> 3.8-Flash (12s Fail-Fast Timeout) ]
                |
                +-- Success ---------> Grounded / Open-Domain Response
                |
                +-- Timeout / Error -> [ Tier 3: Zero-Key Wikipedia Fallback ]


DETAILED ROUTING PIPELINE
================================================================================

1. Tier 1 — Deterministic Local NLP Router (core/nlp_router.py)

   - Vector Space Classification: Fits a Scikit-Learn TfidfVectorizer on
     curated intent patterns in data/intents.json and computes Cosine
     Similarity at inference time (confidence threshold = 0.62).
   - Zero-Cost Execution: Resolves routine conversational intents (greetings,
     system identity, capabilities, current local time and date) directly in
     server memory in 5 to 10 milliseconds without consuming external API quota.
   - Anti-Collision Intent Guards: Enforces a maximum length check (6 words or
     fewer) and a document-keyword filter ("document", "pdf", "file",
     "summarize", "page", "uploaded") so document-targeted or complex queries
     never collide with local intents.


2. Tier 2 — Document RAG and Multi-Model Gemini Cascade (core/rag_engine.py, core/llm_client.py)

   - In-Memory Document Ingestion: Accepts PDF and plain-text uploads via
     multipart form streaming, extracts page-indexed text using pypdf, and
     partitions content into overlapping 180-word segments (30-word overlap) to
     preserve semantic continuity across chunk boundaries.
   - Scoped Context Retrieval: Vectorizes document chunks with English stop-word
     filtering and retrieves top-scoring passages (score > 0.05) with explicit
     page citations. General queries bypass document injection entirely unless
     keywords match or the user explicitly requests a document summary.
   - Multi-Model Circuit Breaker: Cascades automatically across
     gemini-3.5-flash, gemini-3.1-flash-lite, and gemini-3.8-flash using a
     12-second fail-fast HTTP timeout and a 450-token output cap to prevent
     hanging requests during upstream API congestion.
   - Session-Scoped Memory: Maintains a sliding window of recent user-assistant
     exchanges inside Flask server sessions for coherent multi-turn follow-ups.


3. Tier 3 — Zero-Key Web Fallback (core/web_fallback.py)

   - Graceful Degradation: If all Tier 2 generative models experience rate limits,
     network timeouts, or missing API credentials, the router automatically
     falls back to the public Wikipedia REST API to retrieve concise factual
     summaries without requiring an API key.


KEY FEATURES AND ENGINEERING HIGHLIGHTS
================================================================================

- Real-Time Routing Telemetry: Every assistant response displays a live
  telemetry badge indicating the active execution tier (Tier 1, Tier 2 + RAG,
  Tier 2, or Tier 3), similarity confidence score, and end-to-end server
  latency in milliseconds (ms).

- Strict Context Isolation: Separates general knowledge queries from uploaded
  document context so uploading a file never pollutes unrelated conversations
  or local time/date lookups.

- Structured Markdown Presentation: Parses headings, bold emphasis, bulleted
  lists, and fenced code blocks cleanly in the browser via marked.js, paired
  with one-click demonstration prompt chips for rapid evaluation.

- Production Deployment Configuration: Packaged with Gunicorn WSGI server
  bindings (Procfile), dynamic environment port resolution, and strict .env
  credential isolation via .gitignore.


TECHNOLOGY STACK
================================================================================

- Backend Framework        : Python 3, Flask, Gunicorn, Python-Dotenv
- Machine Learning & NLP   : Scikit-Learn (TfidfVectorizer, cosine_similarity), NumPy
- Generative AI & Parsing  : Google GenAI SDK (google-genai), PyPDF (pypdf)
- Frontend Interface       : HTML5, CSS3, Vanilla JavaScript, Marked.js
- Cloud Hosting            : Render Web Service (Linux / Gunicorn)


PROJECT STRUCTURE
================================================================================

    nexusai-assistant/
    ├── app.py                  Flask application, routing pipeline, and session state
    ├── core/
    │   ├── nlp_router.py       Tier 1: TF-IDF intent classifier with RAG keyword guards
    │   ├── rag_engine.py       PDF/TXT ingestion, overlapping chunking, and retrieval
    │   ├── llm_client.py       Tier 2: Gemini Flash cascade client with fail-fast timeouts
    │   └── web_fallback.py     Tier 3: Zero-key Wikipedia REST API fallback handler
    ├── data/
    │   └── intents.json        Training patterns and templates for local Tier 1 intents
    ├── templates/
    │   └── index.html          Responsive chat UI, Markdown renderer, and demo chips
    ├── static/
    │   └── style.css           Dark-mode layout, custom scrollbars, and telemetry pills
    ├── requirements.txt        Pinned Python package dependencies
    ├── Procfile                Gunicorn production process declaration
    └── .gitignore              Excludes .env, venv/, and Python bytecode caches


API ENDPOINTS
================================================================================

1. GET /
   Renders the primary NexusAI Assistant web interface and initializes the
   user session.

2. POST /upload
   Accepts a multipart/form-data file upload (.pdf or .txt), extracts text,
   builds the TF-IDF chunk index in memory, and returns the indexed chunk count.

3. POST /chat
   Accepts a JSON payload {"message": "<user_query>"}, executes the three-tier
   routing pipeline, updates session history, and returns a JSON object
   containing reply, tier, confidence, and latency_ms.


LOCAL SETUP AND INSTALLATION
================================================================================

1. Clone the Repository

   git clone https://github.com/himani01feb/nexusai-assistant.git
   cd nexusai-assistant


2. Create and Activate a Virtual Environment

   Windows:
   python -m venv venv
   venv\Scripts\activate

   macOS / Linux:
   python3 -m venv venv
   source venv/bin/activate


3. Install Required Dependencies

   pip install -r requirements.txt


4. Configure Environment Variables

   Create a .env file in the root directory and add your credentials:

   GEMINI_API_KEY=your_google_gemini_api_key_here
   FLASK_SECRET_KEY=your_random_secret_key_here


5. Launch the Development Server

   python app.py

   Navigate to http://127.0.0.1:5000 in your browser.


DEPLOYMENT GUIDE (RENDER)
================================================================================

1. Push the complete repository to GitHub (confirm that .env is excluded by
   .gitignore before pushing).

2. Log in to Render, select New -> Web Service, and connect your GitHub
   repository.

3. Configure the build and start commands:
   - Runtime       : Python 3
   - Build Command : pip install -r requirements.txt
   - Start Command : gunicorn app:app

4. Open the Environment tab in Render, add GEMINI_API_KEY and FLASK_SECRET_KEY,
   and trigger a manual or automatic deployment.


AUTHOR
================================================================================

Himani Goel
B.Tech in Artificial Intelligence and Data Engineering (AIDE)
Indian Institute of Information Technology Kota (IIIT Kota)
