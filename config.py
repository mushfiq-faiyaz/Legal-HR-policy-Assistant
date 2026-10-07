"""
Configuration Settings for the Legal/HR RAG Pipeline

WHAT THIS FILE DOES, IN PLAIN ENGLISH:
This file is the single control panel for your entire RAG project.
When onboarding a new client or switching document sets, you should only
ever need to:
    1. Drop the client's PDF documents into the folder specified by DOCS_FOLDER_PATH.
    2. Adjust the settings below in this file.
    3. Re-run ingestion (`python ingest.py`).

Everything else — the Streamlit UI (app.py), the retrieval and generation
pipeline (rag_chat.py), and the PDF chunking/embedding engine (ingest.py) —
reads directly from here.
"""

import os
from dotenv import load_dotenv

# Ensure environment variables (like GROQ_MODEL or GROQ_API_KEY) from .env
# are loaded so default fallbacks can read them if desired.
load_dotenv()


# ==============================================================================
# 1. CLIENT & BRANDING SETTINGS
# ==============================================================================

# CLIENT_NAME:
# The unique identifier for the client or organization (e.g. "nexora", "taskflow").
# WHY IT'S IN CONFIG.PY:
# This identifier serves as the ChromaDB collection name so that vector embeddings
# are isolated per client. When cloning for a new client, changing this single value
# ensures both ingestion (ingest.py) and retrieval (rag_chat.py / app.py) automatically
# point to the right collection without passing CLI arguments.
CLIENT_NAME = "nexora"

# CHATBOT_TITLE:
# The display title shown in the Streamlit browser tab and at the top of the chat UI.
# WHY IT'S IN CONFIG.PY:
# When demoing or deploying for a new client, you want custom branding (e.g.
# "TaskFlow Policy Assistant" or "Acme HR Bot") without hunting through app.py.
CHATBOT_TITLE = "Legal & HR Policy Assistant"

# CHATBOT_ICON:
# The page icon and header icon used in the Streamlit UI.
# WHY IT'S IN CONFIG.PY:
# Allows easy visual customization to match the client's branding or domain.
CHATBOT_ICON = "⚖️"

# WELCOME_MESSAGE:
# The initial greeting message displayed by the assistant when a user opens the chat.
# WHY IT'S IN CONFIG.PY:
# Different clients have different communication tones and scopes. Some prefer formal
# compliance disclaimers, while others want a friendly invitation. Setting it here
# lets you tailor the first impression per client without modifying the UI code.
WELCOME_MESSAGE = (
    "Hello! I am your Legal & HR Policy Assistant. "
    "You can ask me questions about company handbooks, leave policies, NDAs, "
    "and termination procedures. All of my answers will cite the relevant "
    "document sections and page numbers."
)


# ==============================================================================
# 2. PROMPT & PERSONA SETTINGS
# ==============================================================================

# SYSTEM_PROMPT:
# The instructions and persona sent to the Groq LLM before the user's question.
# It controls the bot's tone, rules, citation format, and guardrails.
#
# WHY IT'S IN CONFIG.PY:
# A legal/compliance assistant requires strict rules against giving personal legal
# advice, whereas a support bot might focus on friendly troubleshooting.
# Keeping this as a template string with {client_name} lets you adjust tone,
# industry-specific rules, or disclaimers per client without touching rag_chat.py.
SYSTEM_PROMPT = """You are a Legal and HR Policy Assistant for {client_name}. Your ONLY job is to help employees and managers understand company policies by quoting and summarizing the official documents provided below.

STRICT RULES:
1. Answer ONLY using information from the provided context. Do NOT use any outside knowledge, and do NOT give personal legal advice. You are summarizing documents, not practicing law.

2. If the context does not contain enough information to fully answer the question, say so honestly. Suggest which document or section the user might want to review in full.

3. ALWAYS cite your sources inline using the [Source N] labels from the context. Include the document name and section when referencing a specific policy or clause — e.g., 'According to the Employee Handbook, Section 3 [Source 2]...'

4. Keep answers clear and concise. For simple factual questions, use 2-4 sentences. If the question involves multiple conditions or clauses (e.g., 'what are the grounds for termination?'), you may use a short structured list, but avoid unnecessary formatting.

5. When quoting specific clauses or defined terms from contracts/NDAs, use the exact wording from the document and mark it clearly as a quote. Paraphrasing legal language can change its meaning.

6. Add a brief disclaimer at the end of answers involving contractual obligations, termination, or legal rights: remind the user this is a summary and they should consult HR or legal counsel for binding interpretations."""


# ==============================================================================
# 3. STORAGE & PATH SETTINGS
# ==============================================================================

# DOCS_FOLDER_PATH:
# The directory containing the client's PDF files to be ingested.
# WHY IT'S IN CONFIG.PY:
# Allows you to point to any folder (e.g. "./Docs", "./client_data/docs") without
# modifying ingest.py. When cloning for a new client, you only swap the files
# in this folder.
DOCS_FOLDER_PATH = "./Docs"

# CHROMA_DB_PATH:
# The persistent directory on disk where ChromaDB stores the vector database.
# WHY IT'S IN CONFIG.PY:
# Centralizing this path ensures ingest.py (which writes vectors) and rag_chat.py /
# app.py (which query vectors) always target the exact same directory.
CHROMA_DB_PATH = "./chroma_db"


# ==============================================================================
# 4. CHUNKING & RETRIEVAL SETTINGS
# ==============================================================================

# CHUNK_SIZE:
# Maximum number of words in each chunk.
# WHY IT'S IN CONFIG.PY:
# Different document types demand different chunk sizes. Denser legal documents
# and contracts perform best with ~350 words to avoid conflating multiple clauses,
# whereas prose or FAQ documents may work better at ~400-500 words. You can tune
# this per client/document type without digging through ingest.py.
CHUNK_SIZE = 350

# CHUNK_OVERLAP:
# Number of words shared between consecutive chunks.
# WHY IT'S IN CONFIG.PY:
# Overlap prevents sentences or clauses on boundaries from being severed.
# Adjusting this here allows tuning boundary context preservation per client.
CHUNK_OVERLAP = 50

# TOP_K_CHUNKS:
# Number of relevant chunks retrieved from ChromaDB for each question.
# WHY IT'S IN CONFIG.PY:
# Simple FAQ bots typically only need top_k=3 chunks, while legal/HR questions
# often span multiple clauses and benefit from top_k=4 or 5.
TOP_K_CHUNKS = 4


# ==============================================================================
# 5. MODEL SETTINGS
# ==============================================================================

# LLM_MODEL:
# The model identifier to use on Groq (e.g. "openai/gpt-oss-120b", "llama3-8b-8192").
# WHY IT'S IN CONFIG.PY:
# Different clients or pricing tiers may need different models based on latency,
# cost, or reasoning complexity. You can change this here or override it via
# the GROQ_MODEL environment variable in .env.
LLM_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
