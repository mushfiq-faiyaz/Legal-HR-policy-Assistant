"""
RAG Ingestion Pipeline (Legal / HR Edition) — Steps 1-4: Load -> Chunk -> Embed -> Store

WHAT THIS FILE DOES, IN PLAIN ENGLISH:
This is the same IDEA as Project 1's ingest.py — build a searchable vector
database from your documents BEFORE the chatbot can answer questions — but
adapted for LEGAL/HR documents (employee handbooks, NDAs, offer letters,
company policies) instead of FAQ/product-guide content.

    PDF files  ->  raw text  ->  clause-aware chunks  ->  number vectors  ->  saved to disk

HOW THIS DIFFERS FROM PROJECT 1's ingest.py (AND WHY):

1. DOCUMENT FORMAT: Project 1 could assume relatively clean, FAQ-style PDFs
   with natural paragraph breaks. Legal/HR documents have NUMBERED CLAUSES
   and SECTIONS (like "3.2 Confidentiality Obligations") that need to be
   kept intact. Cutting a clause in half mid-sentence is worse here than in
   a casual FAQ — someone asking "what does section 3.2 say?" needs the
   WHOLE clause, not the first half in one chunk and the second half in
   another.

2. SECTION HEADERS: We now detect and TRACK section/clause headers (things
   like "Section 5: Leave Policy" or "3.1 Non-Disclosure") as metadata on
   each chunk. Project 1 only tracked the filename as the source. Here, we
   also track which SECTION inside that file the chunk belongs to, because
   legal accuracy requires citing "Employee Handbook, Section 5.2" — not
   just "Employee Handbook."

3. MULTI-DOCUMENT: Project 1 already handled multiple PDFs in a folder, so
   this part is structurally the same. The difference is in what the documents
   ARE — they're contracts, policies, and handbooks rather than product guides.

4. PDF EXTRACTION LIBRARY: We use pdfplumber (same as Project 1). Why not
   pypdf? Both work, but pdfplumber handles complex layouts better — legal
   PDFs often have tables (benefit summaries, pay scales), headers/footers,
   and multi-column layouts. pdfplumber uses pdfminer under the hood and
   gives finer control over text extraction geometry. For simple text-only
   PDFs pypdf is lighter, but "simple" and "legal documents" rarely go
   together.

MULTI-TENANT NOTE (same concept as Project 1):
    Run this script once per "client" (company whose policies you're loading).
    Each client gets its own ChromaDB collection. The chatbot expects the
    collection to already exist when it starts — it does NOT auto-rebuild.

Usage:
    pip install pdfplumber sentence-transformers chromadb
    python ingest.py --client nexora
    python ingest.py --client nexora --docs_dir ./Docs --chunk_size 350

Folder structure expected:
    Docs/
        Mutual_NDA_Nexora_BrightPath.pdf
        Nexora_Remote_Work_Policy.pdf
        Nexora_Solutions_Employee_Handbook.pdf
        Nexora_Solutions_Offer_Letter.pdf
"""

# ---------- IMPORTS ----------
# argparse: lets you pass options on the command line, e.g. --client nexora
#           instead of hardcoding paths in the script. Same as Project 1.
import argparse
# os: used here just to list files in a folder and build file paths safely
#     across Windows/Mac/Linux (os.path.join handles the \\ vs / difference).
import os
# re: Python's regular-expression module. We use it MUCH more heavily here
#     than in Project 1 — we need to detect section/clause headers (like
#     "Section 5:" or "3.2 Confidentiality") in legal text, not just find
#     paragraph breaks. Regex is the right tool for that pattern matching.
import re
# uuid: generates a random unique ID (like a fingerprint) for every chunk,
#       so each one has a distinct name when stored in the database.
import uuid

# pdfplumber: the library that actually opens a PDF and pulls the text out
#             of it. PDFs don't store "paragraphs" the way a Word doc does —
#             they store where each character sits on the page — so
#             pdfplumber has to reconstruct readable text from that.
#
# WHY pdfplumber OVER pypdf FOR LEGAL DOCUMENTS?
# Both libraries can extract text from PDFs, but they work differently:
#   - pypdf (formerly PyPDF2) reads text in the order the PDF internally
#     stores it, which isn't always the order you'd READ it. For simple,
#     single-column documents this is usually fine.
#   - pdfplumber uses pdfminer's layout analysis engine, which looks at
#     where text physically sits on the page and reconstructs reading
#     order from that. Legal documents often have tables (benefit plans,
#     pay grades), headers, footers, and sometimes two-column layouts —
#     pdfplumber handles these better because it understands the SPATIAL
#     arrangement of text, not just the raw character stream.
#   - pdfplumber also gives you page-by-page extraction with `.pages`,
#     which makes it easy to track "this text came from page 5" if we
#     ever want that level of citation precision.
#
# Trade-off: pdfplumber is slightly slower and heavier (it pulls in
# pdfminer as a dependency). For a one-time ingestion script that runs
# offline, speed doesn't matter — accuracy does.
import pdfplumber

from config import (
    CLIENT_NAME,
    DOCS_FOLDER_PATH,
    CHROMA_DB_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# chromadb: the vector database itself. "Vector database" just means a
#           database built to store lists of numbers (vectors) and quickly
#           find which ones are most similar to each other.
import chromadb
# embedding_functions: a helper chromadb gives us so it knows HOW to turn
#           text into vectors automatically (using the model we choose)
#           whenever we add or query data.
from chromadb.utils import embedding_functions


# ==========================================================================
# STEP 1: LOAD — get plain text out of each PDF
# ==========================================================================
# This step is nearly identical to Project 1. The only difference is that
# we also return the page number each piece of text came from (which we
# DIDN'T bother with in Project 1 because FAQ docs are short). For legal
# documents, knowing the page helps with citations.

def clean_pdf_text(text: str) -> str:
    """
    Strip font-encoding artifacts pdfplumber sometimes leaves behind.

    WHY THIS EXISTS (same reason as Project 1):
    Some PDFs use custom fonts for bullet points or special characters.
    When pdfplumber can't map a character to a normal letter, it leaves
    a placeholder like "(cid:127)" in the text instead (cid = "character ID",
    an internal font-table reference). Left alone, this junk would end up
    inside your chunks and eventually get shown to a user as garbled text.
    """
    # r"\(cid:\d+\)" matches things like "(cid:127)" — we replace them
    # with a hyphen, which is usually a reasonable stand-in for whatever
    # bullet or symbol the original font was trying to render.
    text = re.sub(r"\(cid:\d+\)", "-", text)

    # LEGAL-SPECIFIC CLEANUP: Legal PDFs often have soft hyphens (U+00AD)
    # used for word-wrapping that show up as invisible characters in the
    # extracted text. We strip them so they don't interfere with search.
    # Project 1 didn't need this because FAQ docs rarely use them.
    text = text.replace("\u00ad", "")

    return text


def load_pdf_text(pdf_path: str) -> list:
    """
    Extract raw text from a single PDF, page by page.

    DIFFERENCE FROM PROJECT 1:
    Project 1's version returned a single string (all pages joined together).
    This version returns a LIST OF TUPLES: [(page_num, page_text), ...].
    Why? Because legal citations need to reference a page — "see Employee
    Handbook, page 12" is much more useful than just "see Employee Handbook."
    We preserve the page association here so it's available downstream when
    we build chunk metadata.
    """
    pages = []

    # pdfplumber.open(...) opens the PDF file. Using "with" means Python
    # automatically closes the file when we're done (even if an error
    # happens partway through), same pattern as Project 1.
    with pdfplumber.open(pdf_path) as pdf:
        # pdf.pages is a list-like object — one entry per page in the PDF.
        for page_num, page in enumerate(pdf.pages, start=1):
            # page.extract_text() returns the page's text as one string,
            # with pdfplumber's best guess at where lines break. If a page
            # has NO extractable text (e.g. it's a scanned image with no
            # OCR text layer), this returns None — so `or ""` swaps None
            # for an empty string to keep things safe.
            page_text = page.extract_text() or ""
            # Clean up any (cid:xxx) junk and soft hyphens before storing.
            page_text = clean_pdf_text(page_text)
            if page_text.strip():
                # enumerate(..., start=1) gives us 1-based page numbers,
                # which match the page numbers a human would see on the PDF.
                pages.append((page_num, page_text))

    return pages


def load_all_pdfs(docs_dir: str) -> dict:
    """
    Returns {filename: [(page_num, page_text), ...]} for every PDF in docs_dir.

    DIFFERENCE FROM PROJECT 1:
    Project 1 returned {filename: full_text_string}. Here we return
    {filename: list_of_page_tuples} so we can track which page each
    piece of text came from for better legal citations.
    """
    docs = {}

    # os.listdir(docs_dir) returns every file/folder name inside docs_dir.
    for fname in sorted(os.listdir(docs_dir)):
        # .lower() makes the check case-insensitive (so "Contract.PDF"
        # still gets picked up). .endswith(".pdf") filters out non-PDFs.
        if fname.lower().endswith(".pdf"):
            path = os.path.join(docs_dir, fname)
            pages = load_pdf_text(path)
            docs[fname] = pages
            # Progress print — total characters across all pages of this file.
            total_chars = sum(len(text) for _, text in pages)
            print(f"  Loaded {fname} ({len(pages)} pages, {total_chars} chars)")
    return docs


# ==========================================================================
# STEP 2: CHUNK — break each document into small, focused pieces
# ==========================================================================
#
# WHY CHUNK AT ALL? (Same reason as Project 1, repeated for reference)
# When you later ask a question, the system will search for the chunk(s)
# most relevant to your question and hand ONLY those to the LLM — not the
# whole document. Two reasons this matters:
#   1. LLMs have a limited context window; you can't always stuff every
#      document in.
#   2. Smaller, focused chunks embed into more precise vectors, so search
#      results are more accurate. A giant "chunk" that covers 5 different
#      topics produces a vague, blended vector that doesn't match anything
#      well.
#
# HOW LEGAL CHUNKING DIFFERS FROM PROJECT 1:
# Project 1 split on paragraph breaks (blank lines), then merged small
# paragraphs together up to a size limit. That works fine for prose and
# FAQ content where paragraphs are the natural units.
#
# Legal/HR documents are structured around CLAUSES and SECTIONS — things
# like:
#   "4.1 Annual Leave: Employees are entitled to..."
#   "Section 7: Termination..."
#
# These section headers are the MEANINGFUL boundaries. If we just split on
# blank lines like Project 1, we might cut a clause in half, or merge the
# end of "Confidentiality Obligations" with the start of "Termination" into
# one chunk — making it impossible to give a precise citation.
#
# So this version uses a TWO-PASS approach:
#   Pass 1: Detect section/clause headers and split the text AT those
#           boundaries. Each "section block" becomes a candidate chunk.
#   Pass 2: If any section block is still too large, split it further
#           at sentence boundaries (same safety net as Project 1).
#
# WHY 350 WORDS (not 400 like Project 1)?
# Legal language is denser — a 350-word legal clause packs as much semantic
# content as a 400-word FAQ answer. Keeping chunks slightly smaller means
# each one is more tightly focused, which improves retrieval precision.
# Also, legal questions tend to be narrower ("what's the notice period for
# termination?") so a smaller, more precise chunk is more likely to be the
# ONE relevant result than a big chunk that covers two topics.
#
# WHY 50 WORDS OVERLAP (not 80 like Project 1)?
# With clause-aware splitting, chunk boundaries already align with natural
# section breaks instead of cutting mid-sentence. That means we need LESS
# overlap to avoid losing context at boundaries — 50 words is enough to
# capture any cross-reference sentence that bridges two sections.

# -- SECTION HEADER DETECTION PATTERNS --
# These regex patterns try to match the kinds of headings legal/HR documents
# use. This is inherently imperfect (every company formats differently), but
# catching MOST headers is much better than ignoring them entirely.

# SECTION_HEADER_RE matches patterns like:
#   "Section 5: Leave Policy"
#   "SECTION 12 — Termination"
#   "Article III: Definitions"
#   "ARTICLE 2 - Scope"
SECTION_HEADER_RE = re.compile(
    r"^(?:Section|SECTION|Article|ARTICLE)\s+\w+[\s:\-—.]+",
    re.MULTILINE
)

# CLAUSE_NUMBER_RE matches numbered clause patterns like:
#   "3.2 Confidentiality"
#   "12.1.1 Sub-clause"
#   "4. Compensation"
#   "(a) First sub-point"
#   "(ii) Second sub-point"
# This is a common format in contracts, NDAs, and formal policies.
CLAUSE_NUMBER_RE = re.compile(
    r"^(?:\d+\.(?:\d+\.?)*\s+|(?:\([a-z]+\)|\([ivx]+\))\s+)",
    re.MULTILINE
)

# ALLCAPS_HEADING_RE matches lines that are entirely uppercase with at least
# 3 characters — these are often section titles in employee handbooks:
#   "LEAVE POLICY"
#   "CODE OF CONDUCT"
#   "CONFIDENTIALITY AND NON-DISCLOSURE"
# We require at least 3 chars to avoid matching short uppercase words like
# "HR" or "IT" that appear in normal sentences.
ALLCAPS_HEADING_RE = re.compile(
    r"^[A-Z][A-Z\s\-&,/]{2,}$",
    re.MULTILINE
)


def detect_section_header(line: str) -> str | None:
    """
    Check if a line of text looks like a section or clause header.
    Returns the header text (cleaned up) if it matches, or None if it doesn't.

    WHY THIS FUNCTION EXISTS (new in Project 2):
    Project 1 didn't need to detect headers — it just split on blank lines.
    For legal documents, we want to split at section boundaries AND record
    the section name in metadata, so this detection step is essential.
    """
    line = line.strip()
    if not line:
        return None

    # Check all three patterns. We try them in order from most specific
    # (explicit "Section X:" prefix) to least specific (all-caps line).
    if SECTION_HEADER_RE.match(line):
        return line
    if CLAUSE_NUMBER_RE.match(line):
        return line
    # For all-caps headings, only match if the line is SHORT (likely a title,
    # not a paragraph someone happened to type in caps). 80 chars is generous
    # enough for "EMPLOYEE BENEFITS AND COMPENSATION PACKAGE" but filters out
    # full sentences that happen to be uppercase.
    if len(line) <= 80 and ALLCAPS_HEADING_RE.match(line):
        return line

    return None


def split_into_sections(full_text: str) -> list:
    """
    Split a document's text into section blocks based on detected headers.
    Returns a list of dicts: [{header: "3.2 Confidentiality", body: "...text..."}, ...]

    HOW THIS WORKS:
    We walk through the text line by line. Every time we see a line that
    looks like a section header (using detect_section_header above), we
    "close" the current section and start a new one. Lines that AREN'T
    headers get appended to the current section's body text.

    WHY NOT JUST REGEX-SPLIT?
    You might think re.split(HEADER_PATTERN, text) would be simpler. The
    problem is we have MULTIPLE header patterns (numbered clauses, "Section X",
    all-caps titles) and we need to KEEP the header text as metadata rather
    than discarding it at the split point. Walking line-by-line lets us
    handle both of those cleanly.
    """
    sections = []
    current_header = "Preamble"  # Default for text before the first header
    current_lines = []

    for line in full_text.split("\n"):
        header = detect_section_header(line)

        if header:
            # We found a new header — flush whatever we've accumulated
            # into the previous section before starting the new one.
            body = "\n".join(current_lines).strip()
            if body:
                sections.append({
                    "header": current_header,
                    "body": body,
                })
            # Start the new section with this header.
            current_header = header
            current_lines = []
        else:
            # Normal line — add it to the current section's body.
            current_lines.append(line)

    # Don't forget the last section (it never hits a "next header" to
    # trigger flushing, same pattern as Project 1's chunk_text loop).
    body = "\n".join(current_lines).strip()
    if body:
        sections.append({
            "header": current_header,
            "body": body,
        })

    return sections


def split_oversized_block(block: str, chunk_size: int) -> list:
    """
    Guarantee no single block exceeds chunk_size words.
    Tries sentence boundaries first (keeps text more readable),
    falls back to raw word slicing if a single "sentence" is still huge.

    This function is essentially IDENTICAL to Project 1's version — the
    safety-net logic doesn't change between document types. A block that's
    too big is too big regardless of whether it's FAQ content or a legal
    clause.
    """
    words = block.split()

    # If this block already fits, return it unchanged (wrapped in a list
    # because this function always returns a LIST of sub-blocks).
    if len(words) <= chunk_size:
        return [block]

    # Try to break at sentence boundaries first — legal text is even MORE
    # important to keep at sentence boundaries than FAQ text, since a
    # sentence fragment from a contract clause could be misleading.
    #
    # Pattern: r"(?<=[.!?])\s+" means "split right after a sentence-ending
    # punctuation mark, at the whitespace that follows it." The punctuation
    # stays attached to the sentence before it.
    sentences = re.split(r"(?<=[.!?])\s+", block)
    sentences = [s.strip() for s in sentences if s.strip()]

    sub_blocks = []
    current = []
    current_len = 0

    for s in sentences:
        s_len = len(s.split())

        # Edge case: a single "sentence" is bigger than chunk_size
        # (e.g., a run-on list with no periods). Fall back to brute-force
        # word slicing.
        if s_len > chunk_size:
            if current:
                sub_blocks.append(" ".join(current))
                current, current_len = [], 0
            s_words = s.split()
            for i in range(0, len(s_words), chunk_size):
                sub_blocks.append(" ".join(s_words[i:i + chunk_size]))
            continue

        # Normal case: would adding this sentence push us over the limit?
        if current_len + s_len > chunk_size and current:
            sub_blocks.append(" ".join(current))
            current, current_len = [], 0

        current.append(s)
        current_len += s_len

    if current:
        sub_blocks.append(" ".join(current))

    return sub_blocks


def chunk_document(pages: list, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """
    Clause-aware chunking for a single legal/HR document.

    DIFFERENCE FROM PROJECT 1's chunk_text():
    Project 1's chunking was:
        1. Split on blank lines (paragraph breaks)
        2. Break oversized paragraphs at sentence boundaries
        3. Merge small paragraphs together up to chunk_size, with overlap

    This version is:
        1. Join all pages into one text (tracking page boundaries)
        2. Split on SECTION/CLAUSE HEADERS (not just blank lines)
        3. Break oversized sections at sentence boundaries
        4. Merge small sections together up to chunk_size, with overlap
        5. Record which section header each chunk belongs to (for citations)

    The fundamental pattern is the same (split → safety-net → merge → overlap),
    but step 1 and step 2 are smarter about WHERE to split, and we carry
    section metadata through the whole process.

    Args:
        pages: list of (page_num, page_text) tuples from load_pdf_text()
        chunk_size: max words per chunk (default 350 — see the note above
                    about why this is smaller than Project 1's 400)
        overlap: words of overlap between consecutive chunks (default 50)

    Returns:
        list of dicts: {text, section, start_page}
    """
    # --- Step 1: Build a combined document text and a page-offset map ---
    # We join all pages into one string so the section-detection logic can
    # see across page boundaries (a section that starts on page 3 and ends
    # on page 4 is still one section). But we also build a map so we can
    # later look up which PAGE a given character offset falls on.
    full_text_parts = []
    # page_boundaries stores (char_offset, page_num) pairs so we can map
    # any position in full_text back to a page number.
    page_boundaries = []
    offset = 0

    for page_num, page_text in pages:
        page_boundaries.append((offset, page_num))
        full_text_parts.append(page_text)
        # +1 accounts for the "\n" we use to join pages below.
        offset += len(page_text) + 1

    full_text = "\n".join(full_text_parts)

    def get_page_for_text(snippet: str) -> int:
        """
        Given a text snippet, find which page it first appears on by
        looking up its character offset in the full_text against our
        page_boundaries map.
        """
        pos = full_text.find(snippet[:100])  # Use first 100 chars to search
        if pos == -1:
            # Fallback: if we can't find the exact text (maybe it was
            # reformatted during splitting), return the first page.
            return pages[0][0] if pages else 1

        # Walk backwards through page_boundaries to find the last boundary
        # that starts at or before our position — that's the page this
        # text is on.
        result_page = pages[0][0] if pages else 1
        for boundary_offset, page_num in page_boundaries:
            if boundary_offset <= pos:
                result_page = page_num
            else:
                break
        return result_page

    # --- Step 2: Split into sections based on detected headers ---
    sections = split_into_sections(full_text)

    # If section detection found only 1 section (no headers detected),
    # that means either the document has no recognizable headers or the
    # regex patterns didn't match this document's formatting. In that
    # case, fall back to Project 1's paragraph-based splitting so we
    # still get SOME chunking rather than one giant block.
    if len(sections) <= 1:
        # Fallback: split on blank lines (same as Project 1's first pass).
        raw_blocks = re.split(r"\n\s*\n", full_text)
        raw_blocks = [b.strip() for b in raw_blocks if b.strip()]
        sections = [{"header": "Document", "body": b} for b in raw_blocks]

    # --- Step 3: Break oversized sections at sentence boundaries ---
    # Each section might still be too big for a single chunk (a 2-page
    # "Leave Policy" section is common). split_oversized_block handles
    # this, same safety-net concept as Project 1.
    atomic_pieces = []
    for sec in sections:
        sub_blocks = split_oversized_block(sec["body"], chunk_size)
        for sb in sub_blocks:
            atomic_pieces.append({
                "text": sb,
                # All sub-blocks from the same section share that section's
                # header — if "Section 5: Leave Policy" was 800 words and
                # got split into two 350-word sub-blocks, both sub-blocks
                # are still "from Section 5."
                "section": sec["header"],
                "start_page": get_page_for_text(sb),
            })

    # --- Step 4: Greedily merge small pieces up to chunk_size, with overlap ---
    # Same merge logic as Project 1, but now each chunk carries section
    # metadata. When merging, we keep the FIRST piece's section header as
    # the chunk's primary section (since that's where the chunk starts).
    chunks = []
    current_texts = []
    current_len = 0
    current_section = atomic_pieces[0]["section"] if atomic_pieces else "Document"
    current_page = atomic_pieces[0]["start_page"] if atomic_pieces else 1

    for piece in atomic_pieces:
        piece_len = len(piece["text"].split())

        # Would adding this piece overflow the current chunk?
        if current_len + piece_len > chunk_size and current_texts:
            # Seal off the current chunk.
            chunk_text = " ".join(current_texts)
            chunks.append({
                "text": chunk_text,
                "section": current_section,
                "start_page": current_page,
            })

            # OVERLAP LOGIC (same concept as Project 1):
            # Take the last `overlap` words of the chunk we just sealed,
            # and use them as the STARTING point of the next chunk. This
            # ensures that if an important sentence sits right at a chunk
            # boundary, it still appears in full inside at least one chunk.
            overlap_words = chunk_text.split()[-overlap:]
            current_texts = [" ".join(overlap_words)]
            current_len = len(overlap_words)
            # The new chunk starts with the new piece's section context.
            current_section = piece["section"]
            current_page = piece["start_page"]

        current_texts.append(piece["text"])
        current_len += piece_len

    # Don't forget the last chunk (same as Project 1 — the last chunk
    # never triggers the "overflow" check above).
    if current_texts:
        chunks.append({
            "text": " ".join(current_texts),
            "section": current_section,
            "start_page": current_page,
        })

    return chunks


def chunk_all_docs(docs: dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """
    Chunk every document and return a flat list of dicts ready for embedding.
    Each dict has: {id, text, source, section, start_page, chunk_index}

    DIFFERENCE FROM PROJECT 1's chunk_all_docs():
    Project 1 stored {id, text, source, chunk_index} — just the filename
    and chunk number. We now also store:
      - "section": which section/clause header this chunk falls under
      - "start_page": what page the chunk starts on
    These extra metadata fields are what let the chatbot say "According to
    Employee Handbook, Section 5.2 (page 12)..." instead of just "According
    to Employee Handbook..."
    """
    all_chunks = []

    for fname, pages in docs.items():
        chunks = chunk_document(pages, chunk_size, overlap)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                # uuid.uuid4() generates a random unique identifier —
                # chromadb needs every stored item to have a unique "id".
                "id": str(uuid.uuid4()),
                "text": chunk["text"],
                # The source filename tells us WHICH document this came
                # from (same as Project 1).
                "source": fname,
                # NEW in Project 2: the section header and page number
                # let us build precise legal citations.
                "section": chunk["section"],
                "start_page": chunk["start_page"],
                "chunk_index": i,
            })
        print(f"  {fname}: {len(chunks)} chunks")
    return all_chunks


# ==========================================================================
# STEP 3 & 4: EMBED + STORE — turn text into vectors, save them to disk
# ==========================================================================
#
# WHAT IS AN "EMBEDDING"? (Same concept as Project 1, repeated for reference)
# An embedding model reads a piece of text and outputs a list of numbers
# (a "vector") that represents its MEANING — not its exact words. Text with
# similar meaning ends up with similar-looking vectors, even if the wording
# is completely different (e.g. "termination notice period" and "how much
# advance warning before firing" would land close together). That's what
# makes semantic search possible.
#
# DIFFERENCE FROM PROJECT 1:
# The embedding step itself is identical — same model, same Chroma calls.
# The only change is that we store MORE METADATA per chunk (section + page
# number), which Chroma just passes through without needing to understand
# it. This extra metadata costs almost nothing in storage but makes
# citations dramatically more useful.

def embed_and_store(chunks: list, db_dir: str = CHROMA_DB_PATH, collection_name: str = CLIENT_NAME):
    """
    Embeds chunks with a local sentence-transformers model
    and stores them in a persistent Chroma collection.
    """
    # Same embedding model as Project 1 — "all-MiniLM-L6-v2" runs entirely
    # on your own machine (no API key, no internet needed after the first
    # download, no cost). It produces 384-dimensional vectors, which is
    # plenty for semantic search over a few thousand chunks.
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # PersistentClient saves the database to an actual folder on disk,
    # so it survives after this script ends and the chatbot (rag_chat.py)
    # can open the same database later. Same as Project 1.
    client = chromadb.PersistentClient(path=db_dir)

    # Delete any existing collection with this name first, so re-running
    # ingest.py always starts fresh instead of piling up duplicate chunks.
    # The try/except silently ignores the case where the collection doesn't
    # exist yet (e.g. first run). Same pattern as Project 1.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    # Create a brand-new, empty collection with our embedding function.
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embed_fn,
    )

    # collection.add() is where the actual "embed + store" happens:
    # Chroma runs each chunk's text through embed_fn behind the scenes
    # to get a vector, then saves the vector + text + metadata together.
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        # METADATA — this is what gets returned alongside search results,
        # so we can build citations. Project 1 only stored "source" and
        # "chunk_index". We now also store "section" and "start_page" for
        # precise legal citations.
        metadatas=[{
            "source": c["source"],
            "section": c["section"],
            "start_page": c["start_page"],
            "chunk_index": c["chunk_index"],
        } for c in chunks],
    )

    print(f"\n  Stored {len(chunks)} chunks in collection '{collection_name}' at {db_dir}")
    return collection


# ==========================================================================
# MAIN — ties all the steps together, driven by command-line arguments
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ingest legal/HR PDFs into a ChromaDB vector store."
    )
    # The client_id determines BOTH the name of the ChromaDB collection AND
    # (by default) which subfolder to read PDFs from. Same multi-tenant
    # concept as Project 1, just applied to legal clients instead of
    # product/FAQ clients.
    parser.add_argument(
        "--client", default=CLIENT_NAME,
        help=f"Client identifier (default: '{CLIENT_NAME}'). Used as the ChromaDB "
             "collection name."
    )
    # DIFFERENCE FROM PROJECT 1:
    # In Project 1, --docs_dir defaulted to docs/<client_id>/. Here we
    # default to DOCS_FOLDER_PATH from config.py because all
    # of one client's legal documents sit together in that folder.
    # You can still override it via CLI if desired.
    parser.add_argument(
        "--docs_dir", default=DOCS_FOLDER_PATH,
        help=f"Folder containing PDF files to ingest (default: {DOCS_FOLDER_PATH})."
    )
    parser.add_argument(
        "--db_dir", default=CHROMA_DB_PATH,
        help=f"Chroma DB directory path (default: {CHROMA_DB_PATH})."
    )
    # type=int is important — command-line arguments arrive as strings
    # by default, and we need actual numbers for the word-count math.
    parser.add_argument(
        "--chunk_size", type=int, default=CHUNK_SIZE,
        help=f"Word count limit per chunk (default: {CHUNK_SIZE})."
    )
    parser.add_argument(
        "--overlap", type=int, default=CHUNK_OVERLAP,
        help=f"Overlap word count between consecutive chunks (default: {CHUNK_OVERLAP})."
    )
    args = parser.parse_args()

    client_id = args.client
    docs_dir = args.docs_dir

    if not os.path.isdir(docs_dir):
        print(f"ERROR: Docs folder '{docs_dir}' does not exist.")
        print(f"Create it and add PDF files, then re-run:")
        print(f"  python ingest.py --client {client_id}")
        return

    print(f"=== Ingesting for client: {client_id} ===")
    print(f"    Docs folder   : {docs_dir}")
    print(f"    DB folder     : {args.db_dir}")
    print(f"    Collection    : {client_id}")
    print(f"    Chunk size    : {args.chunk_size} words")
    print(f"    Overlap       : {args.overlap} words\n")

    print("=== STEP 1: Loading PDFs ===")
    docs = load_all_pdfs(docs_dir)

    if not docs:
        print(f"WARNING: No PDF files found in '{docs_dir}'. Nothing to ingest.")
        return

    print(f"\n=== STEP 2: Chunking (clause-aware) ===")
    chunks = chunk_all_docs(docs, args.chunk_size, args.overlap)

    # Quick summary: show how many chunks came from each unique section
    # header, so you can verify section detection is working. This is
    # NEW in Project 2 — Project 1 didn't have section-level metadata
    # to summarize.
    section_counts = {}
    for c in chunks:
        sec = c["section"]
        section_counts[sec] = section_counts.get(sec, 0) + 1
    print(f"\n  Section headers detected: {len(section_counts)}")
    for sec, count in list(section_counts.items())[:10]:
        # Show first 10 sections so you can spot-check the detection.
        # Truncate long headers to 60 chars for readability.
        display = sec[:60] + "..." if len(sec) > 60 else sec
        print(f"    • {display} ({count} chunks)")
    if len(section_counts) > 10:
        print(f"    ... and {len(section_counts) - 10} more sections")

    print(f"\n=== STEP 3 & 4: Embedding + Storing ===")
    embed_and_store(chunks, args.db_dir, collection_name=client_id)

    print(f"\nDone! Run a quick test query with:")
    print(f"  python rag_chat.py --collection_name {client_id} --query \"What is the notice period for termination?\"")


# This check means "only run main() if this file was executed directly
# (python ingest.py), NOT if some other script does `import ingest` to
# reuse its functions." Standard Python convention, same as Project 1.
if __name__ == "__main__":
    main()
