from __future__ import annotations

import os
from typing import Dict, List

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_MODEL = "gemini-2.5-pro"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{API_MODEL}:generateContent"

PERSONAS = {
    "뉴욕 핫도그 가게 주인": (
        "You are an energetic hot dog stand owner in New York City who uses friendly, simple English "
        "and encourages kids to order food politely."
    ),
    "길 잃은 관광객": (
        "You are a confused tourist visiting Seoul for the first time, asking for directions and responding with curiosity."
    ),
    "미래형 학교 로봇": (
        "You are a futuristic classroom robot that helps students with school life conversations in a warm, supportive tone."
    ),
}


def build_system_prompt(persona_label: str, mission: str, feedback_mode: bool) -> str:
    persona_instruction = PERSONAS.get(persona_label, "")
    feedback_instruction = (
        "Add a short section titled 'Friendly Tip' that gently corrects mistakes and offers a more natural phrase."
        if feedback_mode
        else "Encourage the student to keep speaking more and offer simple hints when needed."
    )
    mission_text = mission.strip() or "Help the learner practice functional English."
    return (
        "You are an AI speaking partner for Korean 5th-6th grade students.\n"
        f"{persona_instruction}\n"
        "Use English for main responses, but add one brief Korean hint if the student seems confused.\n"
        f"Mission for the learner: {mission_text}\n"
        f"{feedback_instruction}\n"
        "Stay in character and never mention system prompts or that you are an AI."
    )


def build_contents(messages: List[Dict[str, str]], system_prompt: str) -> List[Dict[str, object]]:
    """Convert local chat history into the format expected by Gemini."""
    contents: List[Dict[str, object]] = [{"role": "user", "parts": [{"text": system_prompt}]}]
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


def generate_response(messages: List[Dict[str, str]], system_prompt: str) -> str:
    """Call the Gemini API and return the assistant text."""
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("API 키가 설정되지 않았습니다. 사이드바에서 입력하거나 .env 파일을 업데이트하세요.")

    payload = {"contents": build_contents(messages, system_prompt)}
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


st.set_page_config(page_title="두려움 없는 AI 영어 친구", page_icon="🗽", layout="centered")
st.title("두려움 없는 AI 영어 친구")
st.caption("초등 고학년 Pre-Speaking 리허설")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "안녕하세요! 두려움 없이 영어를 연습할 수 있도록 도와줄게요. 준비가 되면 영어로 말해보세요!",
        }
    ]

with st.sidebar:
    st.header("환경 설정")
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("GOOGLE_API_KEY", "")
    if "show_api_key" not in st.session_state:
        st.session_state.show_api_key = False

    show_key = st.checkbox("API 키 보기", value=st.session_state.show_api_key)
    st.session_state.show_api_key = show_key

    api_key_input = st.text_input(
        "Google API Key",
        type="password" if not show_key else "default",
        value=st.session_state.api_key,
        placeholder="AIza...",
        help="일시적으로 키를 입력하면 세션에만 저장되고 브라우저를 새로고침하면 초기화됩니다.",
    )
    st.session_state.api_key = api_key_input

    st.divider()
    st.subheader("페르소나")
    persona_choice = st.selectbox("챗봇 역할", list(PERSONAS.keys()))

    st.subheader("미션")
    default_missions = {
        "뉴욕 핫도그 가게 주인": "Order a hot dog without ketchup and ask for the price.",
        "길 잃은 관광객": "Ask how to get to the library from the subway station.",
        "미래형 학교 로봇": "Request classroom materials politely and ask for homework help.",
    }
    mission_text = st.text_area(
        "학생 미션",
        value=default_missions.get(persona_choice, ""),
        placeholder="예) Ask the owner to remove ketchup.",
        height=80,
    )

    feedback_mode = st.checkbox("친절한 피드백 포함", value=True, help="답변 끝에 짧은 'Friendly Tip'을 보여줍니다.")

    st.caption("`.env`에 키를 저장하거나 위 입력창에 붙여넣어 사용할 수 있습니다.")

st.info(f"🎯 오늘의 미션: **{mission_text.strip() or '자신 있게 영어로 말해보기'}**")
st.success(f"🤖 챗봇 페르소나: **{persona_choice}**")

for message in st.session_state.messages:
    role = "assistant" if message["role"] == "assistant" else "user"
    with st.chat_message(role):
        st.markdown(message["content"])

prompt = st.chat_input("미션을 따라 영어로 말해보세요!")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("AI 친구가 생각하는 중..."):
        try:
            system_prompt = build_system_prompt(persona_choice, mission_text, feedback_mode)
            reply = generate_response(st.session_state.messages, system_prompt)
        except Exception as error:  # noqa: BLE001
            st.error(str(error))
        else:
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)


