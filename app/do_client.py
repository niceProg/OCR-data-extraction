import os
import requests
from typing import Tuple, Dict, Any

# Use same endpoint the user provided
DO_API_URL = "https://inference.do-ai.run/v1/chat/completions"
DO_API_KEY = os.getenv("DO_MODEL_ACCESS_KEY", "")
MODEL_NAME = os.getenv("MODEL")


class DOClientError(Exception):
    pass


def call_do_gpt4o(
    prompt: str, system: str | None = None, timeout: int = 60
) -> Dict[str, Any]:
    """
    Call DigitalOcean Inference Chat Completions endpoint using the user's snippet format.
    Returns the parsed JSON response (raw).
    """
    if not DO_API_KEY or DO_API_KEY == "YOUR_MODEL_ACCESS_KEY":
        raise DOClientError(
            "DigitalOcean model access key not set. Set DO_MODEL_ACCESS_KEY env var."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DO_API_KEY}",
    }

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        # you can add "temperature", "max_tokens" if supported by provider
    }

    resp = requests.post(DO_API_URL, headers=headers, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise DOClientError(f"DO API error: {e}, body: {resp.text}")

    return resp.json()


def extract_assistant_content(response_json: Dict[str, Any]) -> str:
    """
    Try to extract assistant textual content from common response shapes.
    """
    # OpenAI-like: choices[0].message.content
    if not response_json:
        return ""

    if isinstance(response_json, dict):
        if (
            "choices" in response_json
            and isinstance(response_json["choices"], list)
            and response_json["choices"]
        ):
            ch = response_json["choices"][0]
            if isinstance(ch, dict):
                msg = ch.get("message")
                if isinstance(msg, dict):
                    return msg.get("content", "") or ""
                # older OpenAI style
                return ch.get("text", "") or ""
        # sometimes provider returns 'output' or 'response'
        if "output" in response_json:
            # could be string or list
            out = response_json["output"]
            if isinstance(out, list):
                return " ".join([str(x) for x in out])
            return str(out)
        if "response" in response_json:
            return str(response_json["response"])
    return str(response_json)
