import os
import requests
from typing import Dict, Any

DO_API_URL = "https://inference.do-ai.run/v1/chat/completions"
DO_API_KEY = os.getenv("DO_MODEL_ACCESS_KEY", "")
MODEL_NAME = os.getenv("MODEL", "openai-gpt-4o")


class DOClientError(Exception):
    pass


def call_do_gpt4o(
    prompt: str,
    system: str | None = None,
    file_bytes: bytes | None = None,
    file_type: str | None = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    if not DO_API_KEY or DO_API_KEY == "YOUR_MODEL_ACCESS_KEY":
        raise DOClientError("DigitalOcean model access key not set.")

    headers = {"Authorization": f"Bearer {DO_API_KEY}"}

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Payload dasar
    payload = {"model": MODEL_NAME, "messages": messages}

    if file_bytes:
        files = {
            "file": ("upload.png", file_bytes, file_type or "image/png"),
            "payload_json": (None, str(payload), "application/json"),
        }
        resp = requests.post(DO_API_URL, headers=headers, files=files, timeout=timeout)
    else:
        headers["Content-Type"] = "application/json"
        resp = requests.post(DO_API_URL, headers=headers, json=payload, timeout=timeout)

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise DOClientError(f"DO API error: {e}, body: {resp.text}")

    return resp.json()


def extract_assistant_content(response_json: Dict[str, Any]) -> str:
    if not response_json:
        return ""
    if "choices" in response_json and response_json["choices"]:
        ch = response_json["choices"][0]
        if "message" in ch and isinstance(ch["message"], dict):
            return ch["message"].get("content", "")
        return ch.get("text", "")
    if "output" in response_json:
        return str(response_json["output"])
    if "response" in response_json:
        return str(response_json["response"])
    return str(response_json)
