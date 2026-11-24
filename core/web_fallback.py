import requests


def search_wikipedia(query: str):
    cleaned = query.strip()

    # 1. Try DuckDuckGo Instant Answer API (free, zero API key, handles definitions & facts)
    try:
        ddg_res = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": cleaned, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=4,
        ).json()
        abstract = ddg_res.get("AbstractText")
        if abstract:
            source = ddg_res.get("AbstractSource", "Web")
            return f"**{ddg_res.get('Heading', cleaned)}** (via {source}):\n{abstract}"
    except Exception as e:
        print(f"[Tier 3 DDG Error]: {e}")

    # 2. Try Wikipedia Search + Summary API
    conversational_starts = ("can you", "could you", "help me", "write", "create", "make")
    if cleaned.lower().startswith(conversational_starts):
        return None

    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": cleaned,
            "format": "json",
            "srlimit": 1,
        }
        headers = {"User-Agent": "NexusAI-Assistant/2.0"}
        res = requests.get(search_url, params=params, headers=headers, timeout=4).json()
        results = res.get("query", {}).get("search", [])
        if not results:
            return None

        title = results[0]["title"]
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
        summary_res = requests.get(summary_url, headers=headers, timeout=4).json()
        extract = summary_res.get("extract")
        if extract:
            return f"**{title}** (via Wikipedia Fallback):\n{extract}"
    except Exception as e:
        print(f"[Tier 3 Wikipedia Error]: {e}")

    return None