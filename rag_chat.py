"""
RAG Query Pipeline (Legal / HR Edition) — Steps 5-7: Retrieve -> Generate -> Cite

WHAT THIS FILE DOES, IN PLAIN ENGLISH:
ingest.py built the database. This script is where it finally becomes a real
"chatbot": you ask a question about company policies, and it:

    your question
        --> RETRIEVE the most relevant chunks from ChromaDB
        --> GENERATE an answer by handing those chunks + your question to Groq
        --> CITE which document AND which section/clause the answer came from

This is the "RAG" part of Retrieval-Augmented Generation in full: we're
"augmenting" what the LLM can say by "retrieving" real facts from your
legal/HR docs FIRST, instead of letting it just guess from what it was
trained on. For a legal/HR assistant this is especially critical — you
absolutely do NOT want the bot inventing a termination policy or NDA clause
that doesn't exist in the actual documents.

HOW THIS DIFFERS FROM PROJECT 1's rag_chat.py:
1. CITATIONS: Project 1 cited just the filename: "[Source 1: TaskFlow_FAQ.pdf]".
   This version cites the SECTION and PAGE too: "[Source 1: Employee Handbook |
   Section 3: Employment Basics | p.5]". Legal accuracy demands knowing exactly
   WHERE in a document something was stated.

2. SYSTEM PROMPT: Project 1's prompt was for a friendly "customer support
   assistant." This version's prompt is for a "legal/HR policy assistant" —
   it's stricter about accuracy, warns the user when information might be
   incomplete, and explicitly tells the model to never give legal advice
   (just quote what the documents say).

3. ANSWER FORMAT: Project 1 wanted short chat-bubble answers (2-4 sentences).
   Legal/policy questions often need slightly more detail — the prompt here
   allows structured answers when the question genuinely requires listing
   multiple conditions or clauses, while still discouraging unnecessary
   verbosity.

Usage:
    pip install groq python-dotenv chromadb sentence-transformers

    Get a free API key at console.groq.com (no credit card required),
    then put it in a .env file in this same folder:
        GROQ_API_KEY=your_groq_key_here

    Then run:
        python rag_chat.py --collection_name nexora --query "What is the notice period?"
"""

import argparse
import logging
import os
import time

# load_dotenv() reads a ".env" file in your project folder and copies its
# key=value lines into the environment variables for this Python process.
# WHY DO THIS INSTEAD OF PASTING YOUR KEY DIRECTLY IN THE CODE?
# Same reason as Project 1: if your API key is typed directly in the code
# and you ever push to GitHub, your key leaks with it. The .env file
# (which you add to .gitignore) keeps the secret separate from the code.
from dotenv import load_dotenv

from config import (
    CLIENT_NAME,
    CHROMA_DB_PATH,
    LLM_MODEL,
    SYSTEM_PROMPT,
    TOP_K_CHUNKS,
)

import chromadb
from chromadb.utils import embedding_functions

# The official Groq Python SDK. Groq is a cloud API (not local!) that
# runs open-source models on custom hardware built for speed. It's
# genuinely free — no credit card required — for a generous daily request
# limit. Same provider as Project 1, same reasoning for choosing it.
from groq import Groq

# Module-level logger — logs to the console/server logs, not to the user.
# Using __name__ gives us "rag_chat" as the logger name, so log output
# from this module is clearly labeled.
logger = logging.getLogger(__name__)


def _call_with_retry(fn, description: str, max_retries: int = 3, delay: float = 1.0):
    """
    Call `fn()` up to `max_retries` times. If it raises an exception,
    log the details and wait `delay` seconds before retrying. If all
    attempts fail, raise the last exception.

    IDENTICAL to Project 1's version — retry logic doesn't change between
    document types. Network failures, rate limits, and API hiccups happen
    regardless of whether you're asking about refund policies or NDAs.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            logger.warning(
                "%s failed (attempt %d/%d): %s",
                description, attempt, max_retries, e,
                exc_info=True,  # includes the full traceback in the log
            )
            if attempt < max_retries:
                time.sleep(delay)
            else:
                raise


# ==========================================================================
# STEP 5: RETRIEVE — pull the most relevant chunks out of the vector DB
# ==========================================================================
# This function is structurally the same as Project 1's retrieve_chunks.
# The difference is in what METADATA we extract from each result — Project 1
# only pulled "source" (filename) and "chunk_index". We now also pull
# "section" (the clause/section header) and "start_page" (the page number),
# which were stored during ingestion.

def retrieve_chunks(query: str, db_dir: str, collection_name: str, top_k: int) -> list:
    """
    Search the Chroma database for the top_k chunks most similar in
    meaning to `query`. Returns a list of dicts with text + rich metadata.

    DIFFERENCE FROM PROJECT 1:
    Each returned dict now has: {text, source, section, start_page, chunk_index, distance}
    Project 1 only had: {text, source, chunk_index, distance}
    The extra fields ("section", "start_page") are what make legal-grade
    citations possible.
    """
    # Same embedding model as ingest.py used — this MUST match, or the
    # question's vector and the stored chunks' vectors won't be comparable.
    # (If you used model A to embed the chunks but model B to embed the
    # question, you're comparing apples to oranges — the similarity math
    # would be meaningless.)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(collection_name, embedding_function=embed_fn)

    results = _call_with_retry(
        lambda: collection.query(query_texts=[query], n_results=top_k),
        description="ChromaDB query",
    )

    # Reshape Chroma's "list of lists" result format into a clean list
    # of plain dicts — same transformation as Project 1, just with more
    # metadata fields.
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            # NEW in Project 2: section and page metadata for legal citations.
            "section": meta.get("section", "Unknown Section"),
            "start_page": meta.get("start_page", "?"),
            "chunk_index": meta["chunk_index"],
            "distance": dist,
        })
    return chunks


# ==========================================================================
# STEP 6: GENERATE — hand the retrieved chunks + question to the LLM
# ==========================================================================

def build_context_block(chunks: list) -> str:
    """
    Format retrieved chunks into a single block of text to hand the LLM,
    with clear labels so it knows which chunk came from where.

    DIFFERENCE FROM PROJECT 1:
    Project 1 labeled each chunk as: "[Source 1: TaskFlow_FAQ.pdf]"
    This version labels each chunk as:
        "[Source 1: Employee_Handbook.pdf | Section 3: Employment Basics | Page 5]"

    The richer label gives the LLM the information it needs to cite not
    just the DOCUMENT but the specific SECTION and PAGE — which is what
    makes the difference between "check the handbook" (unhelpful) and
    "see Employee Handbook, Section 3, page 5" (actually useful for legal).
    """
    blocks = []
    for i, c in enumerate(chunks):
        # Build a rich source label that includes document, section, and page.
        # .replace(".pdf", "") strips the file extension from the display
        # name — "Nexora_Solutions_Employee_Handbook" reads better than
        # "Nexora_Solutions_Employee_Handbook.pdf" in a citation.
        doc_name = c["source"].replace(".pdf", "").replace("_", " ")

        # Truncate very long section headers to 80 chars so they don't
        # overwhelm the label. Legal section headers can be quite verbose
        # (e.g., "5.2 Upon the Disclosing Party's request, the Receiving
        # Party shall...") — we want enough to identify it, not the full text.
        section = c["section"]
        if len(section) > 80:
            section = section[:77] + "..."

        label = f"[Source {i + 1}: {doc_name} | {section} | Page {c['start_page']}]"
        blocks.append(f"{label}\n{c['text']}")

    # "---" dividers between chunks make boundaries obvious to the LLM.
    return "\n\n---\n\n".join(blocks)


def build_system_prompt() -> str:
    """
    Format and return the system prompt configured in config.py.

    DIFFERENCE FROM PROJECT 1:
    Project 1's prompt was for a friendly customer support bot for "TaskFlow."
    This prompt is for a LEGAL/HR POLICY ASSISTANT, which has stricter rules:

    1. It must NEVER give legal advice — only quote/summarize what the
       documents say. There's a real liability difference between "the
       handbook says X" and "you should do X."

    2. It must cite the specific SECTION and DOCUMENT, not just the document.
       "See Section 3.2 of the NDA" is useful; "see the NDA" is vague.

    3. It must flag when information might be INCOMPLETE — e.g., if the
       retrieved chunks only partially cover the question, the user needs
       to know "this is what I found, but check the full document to be sure."

    4. Unlike Project 1's strict "2-4 sentences, no bullets" rule, legal
       answers sometimes genuinely NEED structure — listing 3 conditions
       for termination as a paragraph is harder to parse than a short list.
       So we allow structured answers when the content warrants it, but
       still discourage unnecessary verbosity.

    The actual prompt text and client template live in config.py so you can
    freely adjust rules, tone, and industry-specific guardrails per client.
    """
    try:
        return SYSTEM_PROMPT.format(client_name=CLIENT_NAME)
    except (KeyError, IndexError):
        return SYSTEM_PROMPT


def generate_answer(query: str, chunks: list, model: str = LLM_MODEL) -> str:
    """Send the question + retrieved context to Groq and return the answer.

    The model used defaults to LLM_MODEL from config.py (which can be overridden
    by the GROQ_MODEL environment variable or by passing model directly).
    """
    client = Groq()

    context_block = build_context_block(chunks)

    user_message = (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}"
    )

    response = _call_with_retry(
        lambda: client.chat.completions.create(
            model=model,
            max_tokens=700,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": user_message},
            ],
        ),
        description="Groq API call",
    )
    return response.choices[0].message.content


# ==========================================================================
# STEP 7: CITE — tell the user which document(s) + section(s) backed the answer
# ==========================================================================

def format_sources_used(chunks: list) -> str:
    """
    Build a human-readable "Sources" footer listing which documents AND
    SECTIONS were retrieved for this answer.

    DIFFERENCE FROM PROJECT 1:
    Project 1 listed just the filename:
        "- [Source 1] TaskFlow_FAQ.pdf"
    This version lists document + section + page:
        "- [Source 1] Employee Handbook | Section 3: Employment Basics | Page 5"

    This is much more useful for legal/HR queries — if someone asks about
    termination policy and the answer cites "Section 4.2," they can go
    directly to that section in the actual PDF to verify.
    """
    lines = []
    seen = set()
    for i, c in enumerate(chunks):
        # Build a unique key from source + section to avoid duplicate lines.
        # (A single section might supply multiple chunks to the same answer,
        # but we only want to list it once in the footer.)
        key = (c["source"], c["section"])
        if key not in seen:
            doc_name = c["source"].replace(".pdf", "").replace("_", " ")
            section = c["section"]
            if len(section) > 80:
                section = section[:77] + "..."
            lines.append(
                f"- [Source {i + 1}] {doc_name} | {section} | Page {c['start_page']}"
            )
            seen.add(key)
    return "\n".join(lines)


# ==========================================================================
# MAIN — ties retrieve + generate + cite together for one question
# ==========================================================================

def answer_question(
    query: str,
    db_dir: str = CHROMA_DB_PATH,
    collection_name: str = CLIENT_NAME,
    top_k: int = TOP_K_CHUNKS,
) -> dict:
    """
    One-call convenience function: given a question, returns a dict with
    the answer and its sources. This is the function the Streamlit chat
    UI (app.py) will import and call directly — everything above it is
    internal plumbing.

    DIFFERENCE FROM PROJECT 1:
    - Default collection_name is CLIENT_NAME from config.py (defaults to "nexora").
    - Default top_k is TOP_K_CHUNKS from config.py (defaults to 4). We retrieve
      one extra chunk because legal questions often span multiple sections
      (e.g., "what are my leave options?" might involve annual leave, sick leave,
      AND parental leave — three different sections).
    - The returned dict includes richer "sources" text with sections/pages.

    If all retries are exhausted (for either ChromaDB or Groq), this
    catches the final exception and returns a friendly error message
    instead of crashing — the raw exception is logged for debugging.
    """
    try:
        chunks = retrieve_chunks(query, db_dir, collection_name, top_k)
        answer = generate_answer(query, chunks)
        sources = format_sources_used(chunks)
        return {"answer": answer, "sources": sources, "chunks": chunks}
    except Exception as e:
        logger.error("answer_question failed after retries: %s", e, exc_info=True)
        return {
            "answer": (
                "I'm having trouble connecting right now — "
                "please try again in a moment."
            ),
            "sources": "",
            "chunks": [],
        }


def main():
    import sys
    # Ensure Windows console prints UTF-8 characters without cp1252 charmap errors
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')

    # load_dotenv() must run BEFORE we try to use the Groq client,
    # since that's what makes GROQ_API_KEY available. Doing this at the
    # very top of main() avoids a confusing "no API key found" error.
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Ask a question about legal/HR policy documents."
    )
    parser.add_argument(
        "--db_dir", default=CHROMA_DB_PATH,
        help=f"Path to Chroma DB folder (default: {CHROMA_DB_PATH})"
    )
    parser.add_argument(
        "--collection_name", default=CLIENT_NAME,
        help=f"Chroma collection name (default: {CLIENT_NAME})"
    )
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--top_k", type=int, default=TOP_K_CHUNKS,
        help=f"Number of chunks to retrieve (default: {TOP_K_CHUNKS})"
    )
    args = parser.parse_args()

    # Early check: if GROQ_API_KEY isn't set, fail with a clear message
    # instead of letting the Groq SDK throw a cryptic auth error.
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not found.")
        print("Create a .env file in this folder with:")
        print("  GROQ_API_KEY=your_groq_key_here")
        print("Get a free key at: https://console.groq.com")
        return

    print(f"Question: {args.query}\n")

    print("=== STEP 5: Retrieving relevant chunks ===")
    result = answer_question(args.query, args.db_dir, args.collection_name, args.top_k)

    # Show which chunks were retrieved, with their section metadata.
    # This is more verbose than Project 1's output because we want to
    # verify that the right SECTIONS are being found, not just the right files.
    if result["chunks"]:
        for i, c in enumerate(result["chunks"]):
            print(f"  Chunk {i+1}: {c['source']} | {c['section'][:50]} | "
                  f"Page {c['start_page']} | Distance: {c['distance']:.4f}")

    print("\n=== STEP 6: Generated Answer ===")
    print(result["answer"])

    print("\n=== STEP 7: Sources ===")
    print(result["sources"])


if __name__ == "__main__":
    main()
