import requests


def search_wikipedia(query: str):
    # Skip Wikipedia for conversational/personal requests so it never returns random articles
    conversational_starts = ("can you", "could you", "help me", "write", "create", "make", "how do i", "why")
    if query.lower().strip().startswith(conversational_starts):
        return None

    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1,
        }
        headers = {"User-Agent": "HybridAIAssistant/2.0"}
        res = requests.get(search_url, params=params, headers=headers, timeout=5).json()
        results = res.get("query", {}).get("search", [])
        if not results:
            return None

        title = results[0]["title"]
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
        summary_res = requests.get(summary_url, headers=headers, timeout=5).json()
        extract = summary_res.get("extract")
        if extract:
            return f"**{title}** (via Wikipedia Fallback):\n{extract}"
    except Exception as e:
        print(f"[Tier 3 Web Fallback Error]: {e}")
    return None