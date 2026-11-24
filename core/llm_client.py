import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


class GeminiClient:
    def __init__(self, model_name="gemini-3.5-flash"):
        # Fastest models first so Tier 2 responds in 1.5–3s instead of waiting on 3.8-flash
        self.models_to_try = [
            model_name,
            "gemini-3.1-flash-lite",
            "gemini-3.8-flash",
        ]
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = (
            genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=20000),  # 20s fail-fast (above 10s API min)
            )
            if self.api_key
            else None
        )

    def generate(self, user_query: str, history: list, rag_chunks: list = None):
        if not self.client:
            raise RuntimeError("GEMINI_API_KEY is not configured in environment.")

        system_instruction = (
            "You are a fast, concise AI assistant. Give direct, structured answers. "
            "If document excerpts are provided, answer strictly from them and cite page numbers."
        )

        # Keep last 4 turns (2 user + 2 assistant) to minimize input token latency
        formatted_history = ""
        for turn in history[-4:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {turn['content'][:400]}\n"

        rag_context = ""
        if rag_chunks:
            snippets = [f"[Page {c['page']}] {c['text']}" for c in rag_chunks]
            rag_context = "\nDocument Excerpts:\n" + "\n---\n".join(snippets) + "\n"

        full_prompt = (
            f"{rag_context}"
            f"History:\n{formatted_history}"
            f"Question: {user_query}"
        )

        last_err = None
        for model in self.models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3,
                        max_output_tokens=450,  # Caps verbose responses for ~2x faster completion
                    ),
                )
                if response.text:
                    return response.text
            except Exception as e:
                print(f"[Tier 2 Model {model} skipped]: {e}")
                last_err = e

        raise last_err or RuntimeError("All Tier 2 Gemini models failed.")

    