# groq_integration.py
import requests
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cached_groq_answers(query: str, groq_api_key: str, dietary_type: str, goal: str, region: str) -> dict:
    """Fetches Groq answers from multiple models concurrently."""
    logging.info(f"Fetching Groq answers for query: '{query}', pref: '{dietary_type}', goal: '{goal}', region: '{region}'")
    models = ["llama", "mixtral", "gemma"]
    results = {}

    if not groq_api_key:
        return {k: "Groq API key not available." for k in models}

    def groq_call(model_name: str):
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            }
            model_map = {
                "llama": "llama3-70b-8192",
                "mixtral": "mixtral-8x7b-32976",
                "gemma": "gemma2-9b-it"
            }
            prompt = (
                f"User query: '{query}'. "
                f"Give a short and clear {dietary_type} Indian diet tip for {goal}, "
                f"tailored to {region} cuisine. Use local ingredients and be realistic."
            )
            payload = {
                "model": model_map[model_name],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 300
            }
            res = requests.post(url, headers=headers, json=payload, timeout=30)
            res.raise_for_status()
            data = res.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error from {model_name}: {e}"

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(groq_call, m): m for m in models}
        for future in futures:
            model = futures[future]
            try:
                results[model] = future.result()
            except Exception as e:
                results[model] = f"Error: {e}"
    return results
