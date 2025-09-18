import os
import requests
from typing import Dict, Any

# API Config
DO_API_URL = "https://inference.do-ai.run/v1/chat/completions"
DO_API_KEY = os.getenv("DO_MODEL_ACCESS_KEY", "")
MODEL_NAME = os.getenv("MODEL")


class DOClientError(Exception):
    pass


def call_do_gpt4o(
    prompt: str, system: str | None = None, timeout: int = 60
) -> Dict[str, Any]:
    """
    Call DigitalOcean Inference Chat Completions endpoint.
    Returns raw JSON response.
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
    }

    resp = requests.post(DO_API_URL, headers=headers, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise DOClientError(f"DO API error: {e}, body: {resp.text}")

    return resp.json()


def extract_assistant_content(response_json: Dict[str, Any]) -> str:
    """
    Extract assistant textual content from common DO API response shapes.
    """
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
                return ch.get("text", "") or ""
        if "output" in response_json:
            out = response_json["output"]
            if isinstance(out, list):
                return " ".join([str(x) for x in out])
            return str(out)
        if "response" in response_json:
            return str(response_json["response"])
    return str(response_json)


def clean_ocr_text(raw_text: str) -> str:
    """
    Send raw OCR text to GPT-4o via DO API and return cleaned version.
    """
    system_prompt = (
        "You are an assistant that cleans OCR text. "
        "Fix OCR mistakes, spelling, and formatting. "
        "Preserve the original meaning and style. "
        "Output only the cleaned text without extra commentary."
    )

    response = call_do_gpt4o(
        prompt=f"Clean and format this OCR text:\n\n{raw_text}",
        system=system_prompt,
    )

    return extract_assistant_content(response)
