# groq_integration.py
import requests
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cached_groq_answers(query: str, groq_api_key: str, dietary_type: str, goal: str, region: str) -> dict:
    logging.info(f"Fetching Groq answers for query: '{query}', pref: '{dietary_type}', goal: '{goal}', region: '{region}'")
    models = ["llama", "mixtral", "gemma"]
    results = {}
    if not groq_api_key:
        return {k: "Groq API key not available." for k in models}

    def groq_diet_answer_single(model_name: str):
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
            groq_model_map = {
                "llama": "llama3-70b-8192",
                "mixtral": "mixtral-8x7b-32976",
                "gemma": "gemma2-9b-it"
            }
            actual_model_name = groq_model_map.get(model_name.lower(), model_name)
            prompt_content = f"User query: '{query}'. Provide a concise, practical **{dietary_type}** diet suggestion or food item for **{goal}**, tailored for a **{region}** Indian context. Focus on readily available ingredients. Be brief."
            payload = {
                "model": actual_model_name,
                "messages": [{"role": "user", "content": prompt_content}],
                "temperature": 0.5,
                "max_tokens": 250
            }
            logging.info(f"Calling Groq API: {actual_model_name} for query: '{query}' ({dietary_type} {goal}, {region})")
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data and data.get('choices') and data['choices'][0].get('message'):
                return data['choices'][0]['message']['content']
            return f"No suggestion from {actual_model_name} (empty/malformed response)."
        except requests.exceptions.Timeout:
            return f"Timeout error from {model_name}."
        except requests.exceptions.RequestException as e:
            return f"Request error from {model_name}: {e}"
        except Exception as e:
            return f"Error from {model_name}: {e}"

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        future_to_model = {executor.submit(groq_diet_answer_single, name): name for name in models}
        for future in future_to_model:
            model_name = future_to_model[future]
            try:
                results[model_name] = future.result()
            except Exception as e:
                results[model_name] = f"Failed: {e}"
    return results
