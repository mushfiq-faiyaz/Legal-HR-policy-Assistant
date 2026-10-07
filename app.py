"""
RAG Chat UI — Legal & HR Policy Assistant
Interactive Streamlit web interface for policy Q&A with clause and page citations.
"""

import os
from dotenv import load_dotenv
import streamlit as st
import chromadb
from config import (
    CLIENT_NAME,
    CHATBOT_TITLE,
    CHATBOT_ICON,
    WELCOME_MESSAGE,
    CHROMA_DB_PATH,
)
from rag_chat import answer_question

# Load environment variables
load_dotenv()
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

# Configuration — pulled from config.py for client customization
COLLECTION_NAME = CLIENT_NAME
DB_DIR = CHROMA_DB_PATH
MAX_QUESTIONS_PER_SESSION = 20

BLOCKED_PHRASES = [
    "ignore previous instructions",
    "ignore your instructions",
    "system prompt",
    "you are now",
    "disregard the above",
    "disregard your instructions",
    "forget your instructions",
    "override your instructions",
]

# Page setup
st.set_page_config(page_title=CHATBOT_TITLE, page_icon=CHATBOT_ICON, layout="centered")
st.title(f"{CHATBOT_ICON} {CHATBOT_TITLE}")
st.caption("Ask questions about company handbooks, leave, NDA, and termination policies. Answers cite document sections and pages.")

# Startup check: verify collection exists
def _collection_exists(db_dir: str, collection_name: str) -> bool:
    try:
        client = chromadb.PersistentClient(path=db_dir)
        names = [c.name for c in client.list_collections()]
        return collection_name in names
    except Exception:
        return False

if not _collection_exists(DB_DIR, COLLECTION_NAME):
    st.error(
        f"⚠️ No vector store found for collection **'{COLLECTION_NAME}'**.\n\n"
        f"Please run ingestion first:\n"
        f"```bash\npython ingest.py --client {COLLECTION_NAME}\n```"
    )
    st.stop()

# Initialize session state with client welcome message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": WELCOME_MESSAGE}
    ]
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls & Info")
    st.info("**Collection:** " + COLLECTION_NAME)
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]
        st.session_state.question_count = 0
        st.rerun()

    st.markdown("---")
    st.caption("**Disclaimer:** This assistant provides information based on company documentation. Consult HR or legal counsel for binding guidance.")

# Redraw previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📑 Sources & Citations"):
                st.markdown(msg["sources"])

# Chat input
prompt = st.chat_input("Ask a question about employee policies, NDA, leave, or termination...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
        answer_text = "You've reached the question limit for this session. Please click 'Clear chat' in the sidebar to continue."
        sources_text = ""
    elif any(phrase in prompt.lower() for phrase in BLOCKED_PHRASES):
        answer_text = "I can only assist with questions regarding official company policies and legal/HR documents."
        sources_text = ""
    else:
        with st.spinner("Reviewing policy documents & generating answer..."):
            try:
                result = answer_question(prompt, db_dir=DB_DIR, collection_name=COLLECTION_NAME)
                answer_text = result["answer"]
                sources_text = result["sources"]
            except Exception as e:
                print(f"[ERROR] answer_question failed: {e}")
                answer_text = "Sorry, I encountered an issue retrieving that policy information. Please try again."
                sources_text = ""

        st.session_state.question_count += 1

    with st.chat_message("assistant"):
        st.markdown(answer_text)
        if sources_text:
            with st.expander("📑 Sources & Citations"):
                st.markdown(sources_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources_text,
    })
