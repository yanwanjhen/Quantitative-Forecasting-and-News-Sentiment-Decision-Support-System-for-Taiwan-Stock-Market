import os
import json
import time
import requests
import threading
from contextlib import contextmanager
from requests.exceptions import HTTPError

try:
    import streamlit as st
except Exception:
    st = None


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
_request_state = threading.local()
REQUIRE_USER_API_KEY = os.getenv("REQUIRE_USER_API_KEY", "").strip() in {"1", "true", "True", "yes", "YES"}

SYSTEM_PROMPT = (
    "你是一位專業且謹慎的台股投資顧問。"
    "所有回覆必須使用繁體中文，避免簡體中文。"
    "請以台灣投資人容易理解的語氣回答，並避免保證獲利。"
)

@contextmanager
def groq_request_context(api_key=None, model=None):
    previous_api_key = getattr(_request_state, "api_key", None)
    previous_model = getattr(_request_state, "model", None)
    _request_state.api_key = api_key
    _request_state.model = model
    try:
        yield
    finally:
        _request_state.api_key = previous_api_key
        _request_state.model = previous_model


def _extract_json_text(text):
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped

    start_positions = [pos for pos in [stripped.find("{"), stripped.find("[")] if pos != -1]
    end_positions = [pos for pos in [stripped.rfind("}"), stripped.rfind("]")] if pos != -1]
    if start_positions and end_positions:
        start = min(start_positions)
        end = max(end_positions)
        if end > start:
            return stripped[start:end + 1]
    return stripped


class GroqResponse:
    def __init__(self, text):
        self.text = text


class GroqStreamChunk:
    def __init__(self, text):
        self.text = text


class GroqChatModel:
    def __init__(self, model=GROQ_MODEL):
        self.model = model

    def _headers(self):
        api_key = getattr(_request_state, "api_key", None)
        if not api_key and st is not None:
            try:
                if "user_api_key" in st.session_state and st.session_state["user_api_key"]:
                    api_key = st.session_state["user_api_key"]
            except Exception:
                pass

        # Enforce user-supplied keys. We intentionally do not read GROQ_API_KEY from env/secrets.
        if not api_key:
            raise RuntimeError("缺少 API Key。請由使用者在前端輸入後再送出請求。")
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, prompt, stream=False, json_mode=False):
        payload = {
            "model": getattr(_request_state, "model", None) or self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "stream": stream,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def generate_content(self, prompt, generation_config=None, stream=False):
        generation_config = generation_config or {}
        wants_json = generation_config.get("response_mime_type") == "application/json"
        json_mode = wants_json and "JSON Array" not in prompt

        payload = self._payload(prompt, stream=stream, json_mode=json_mode)
        
        max_retries = 3
        for attempt in range(max_retries):
            response = requests.post(
                GROQ_API_URL,
                headers=self._headers(),
                json=payload,
                stream=stream,
                timeout=90,
            )
            
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff (1s, 2s)
                    continue
            
            response.raise_for_status()
            break

        if stream:
            return self._stream_chunks(response)

        content = response.json()["choices"][0]["message"]["content"]
        if wants_json:
            content = _extract_json_text(content)
            json.loads(content)
        return GroqResponse(content)

    def _stream_chunks(self, response):
        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line or not raw_line.startswith(b"data: "):
                continue
            data = raw_line[len(b"data: "):].decode("utf-8", errors="replace")
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
                delta = payload["choices"][0].get("delta", {})
                text = delta.get("content")
                if text:
                    yield GroqStreamChunk(text)
            except Exception:
                continue


model_llm = GroqChatModel()
