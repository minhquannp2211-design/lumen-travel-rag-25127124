"""Lumen Travel: Firebase-protected Streamlit chat with a small JSON RAG guide."""

import time
from pathlib import Path

import streamlit as st
from firebase_client import FirebaseError, create_account, sign_in
from rag import RAGError, answer_question, build_index, load_guide, server_url

st.set_page_config(page_title="Lumen Travel", page_icon="🌴", layout="centered")


def sign_out() -> None:
    """Clear both the login session and the chat history."""
    for key in ("user", "messages"):
        st.session_state.pop(key, None)


@st.cache_resource(show_spinner="Embedding the guide (first run only)...")
def cached_index(data_json: str, url: str):
    """Keyed by JSON text and URL: changed data triggers re-indexing."""
    return build_index(load_guide(data_json), url)


st.title("Lumen Travel")
st.caption("Da Nang travel assistant | Firebase + RAG")

user = st.session_state.get("user")
if user and time.time() >= user["expires_at"]:
    sign_out()
    st.warning("Your session expired. Please sign in again.")
    user = None

if not user:
    st.subheader("Sign in")
    with st.form("login", clear_on_submit=True):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        left, right = st.columns(2)
        with left:
            login_clicked = st.form_submit_button("Sign in", use_container_width=True)
        with right:
            signup_clicked = st.form_submit_button("Create account", use_container_width=True)

    if login_clicked or signup_clicked:
        try:
            if signup_clicked:
                create_account(email, password)
                st.success("Account created. Sign in with your credentials.")
            else:
                st.session_state["user"] = sign_in(email, password)
                st.session_state["messages"] = []
                st.rerun()
        except FirebaseError as exc:
            st.error(str(exc))
    st.stop()  # Prevent anyone not signed in from seeing or using the chat.

with st.sidebar:
    st.write(f"Signed in: **{user['email']}**")
    if st.button("Sign out", use_container_width=True):
        sign_out()
        st.rerun()
    st.caption("All messages are stored only in this Streamlit session.")

st.subheader("Ask about Da Nang")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("hits"):
            with st.expander("Retrieved sources and cosine scores"):
                for hit in message["hits"]:
                    st.markdown(f"**[{hit['id']}] {hit['title']}** — score: `{hit['score']:.4f}`")
                    st.write(hit["text"])
                    st.caption(f"Source: {hit['source']}")

question = st.chat_input("Ask a travel question...")
if question is not None:
    if not question.strip():
        st.warning("Please type a question.")
    else:
        st.session_state["messages"].append({"role": "user", "content": question})
        try:
            url = server_url()
            guide_json = Path(__file__).with_name("knowledge_base.json").read_text(encoding="utf-8")
            index = cached_index(guide_json, url)
            result = answer_question(question, index, url, k=3)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": result["answer"],
                "hits": result["hits"],
            })
        except (RAGError, OSError) as exc:
            st.session_state["messages"].append({
                "role": "assistant", "content": f"Request failed: {exc}"
            })
        st.rerun()
