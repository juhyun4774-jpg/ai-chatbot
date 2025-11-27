from __future__ import annotations

import os
from typing import Dict, List

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_MODEL = "gemini-2.5-pro"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{API_MODEL}:generateContent"


def build_contents(messages: List[Dict[str, str]]) -> List[Dict[str, object]]:
    """Convert local chat history into the format expected by Gemini."""
    contents: List[Dict[str, object]] = []
    for message in messages:
        role = "user" if message["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": message["content"]}]})
    return contents


def get_api_key() -> str | None:
    """Return the active API key from session state or fallback to env."""
    session_key = st.session_state.get("api_key")
    if session_key:
        return session_key.strip()
    return os.getenv("GOOGLE_API_KEY")


def generate_response(messages: List[Dict[str, str]]) -> str:
    """Call the Gemini API and return the assistant text."""
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("API 키가 설정되지 않았습니다. 사이드바에서 입력하거나 .env 파일을 업데이트하세요.")

    payload = {"contents": build_contents(messages)}
    params = {"key": api_key}
    response = requests.post(API_URL, params=params, json=payload, timeout=30)

    if not response.ok:
        raise RuntimeError(f"Gemini API error: {response.status_code} {response.text}")

    data = response.json()
    try:
        candidates = data["candidates"]
        parts = candidates[0]["content"]["parts"]
        text = "\n".join(part.get("text", "") for part in parts).strip()
        if not text:
            raise ValueError("Empty response received.")
        return text
    except (KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(f"Unexpected Gemini payload: {data}") from exc


st.set_page_config(page_title="Gemini Chatbot", page_icon="💬", layout="centered")
st.title("Google Gemini Chatbot")
st.caption("Gemini 1.5 Flash • Streamlit")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "안녕하세요! Google Gemini로 구동되는 챗봇입니다. 무엇을 도와드릴까요?",
        }
    ]

with st.sidebar:
    st.header("환경 설정")
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("GOOGLE_API_KEY", "")

    api_key_input = st.text_input(
        "Google API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="AIza...",
        help="일시적으로 키를 입력하면 세션에만 저장되고 브라우저를 새로고침하면 초기화됩니다.",
    )
    st.session_state.api_key = api_key_input

    st.markdown(
        "- `.env` 파일에 `GOOGLE_API_KEY` 값을 추가하세요.\n"
        "- Gemini 1.5 Flash 모델을 사용합니다."
    )

for message in st.session_state.messages:
    role = "assistant" if message["role"] == "assistant" else "user"
    with st.chat_message(role):
        st.markdown(message["content"])

prompt = st.chat_input("메시지를 입력하세요...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Gemini가 답변을 작성하고 있습니다..."):
        try:
            reply = generate_response(st.session_state.messages)
        except Exception as error:  # noqa: BLE001
            st.error(str(error))
        else:
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)


