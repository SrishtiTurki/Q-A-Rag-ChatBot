from collections import defaultdict
from typing import List, Optional


def build_citation(chunk):
    """Build a clean citation string from chunk data."""
    source = chunk['source'].replace(' [images]', '')
    page = chunk.get('page_number', '?')
    line = chunk.get('line_start', '?')
    
    is_image = '[images]' in chunk['source']
    if is_image:
        return f"({source}, Image content, Page {page})"
    else:
        return f"({source}, Page {page}, Line {line})"


def build_prompt(
    query: str,
    chunks: list,
    partial_match: bool = False,
    pdf_sources: dict = None,
    all_indexed_files: list = None,
    missing_files: list = None
) -> str:

    # Group chunks by source (document name)
    by_source = defaultdict(list)
    for chunk in chunks:
        # Clean source name - remove [images] suffix for grouping
        clean_source = chunk['source'].replace(' [images]', '')
        by_source[clean_source].append(chunk)

    # ─── BUILD PER-DOCUMENT CONTEXT ──────────────────────────────────────
    document_sections = []
    
    for doc_name, doc_chunks in by_source.items():
        # Check if this document has image content
        has_images = any('[images]' in c['source'] for c in doc_chunks)
        
        # Start document section
        section = f"\n{'='*60}\n📄 DOCUMENT: {doc_name}\n{'='*60}\n"
        
        # Add each chunk with citation
        chunk_texts = []
        for chunk in doc_chunks:
            citation = build_citation(chunk)
            text = chunk['text']
            if len(text) > 600:
                text = text[:600] + "..."
            chunk_texts.append(f"{citation}\n{text}")
        
        section += "\n\n".join(chunk_texts)
        document_sections.append(section)

    context_str = "\n".join(document_sections)

    # ─── TRACK MISSING DOCUMENTS ──────────────────────────────────────────
    if missing_files is None:
        if all_indexed_files:
            hit_bases = {c['source'].replace(' [images]', '') for c in chunks}
            missing_files = [f for f in all_indexed_files if f not in hit_bases]
        else:
            missing_files = []

    # ─── DETECT QUERY TYPE ──────────────────────────────────────────────────
    is_summary = any(
        w in query.lower()
        for w in ["summarize", "summary", "summarise", "overview", "brief", 
                  "what is in", "what does", "tell me about", "describe", 
                  "what are", "what's in"]
    )

    # ─── SUMMARY PROMPT ──────────────────────────────────────────────────────
    if is_summary:
        # Build the missing documents note
        missing_note = ""
        if missing_files:
            missing_note = "\n\n⚠️ Documents with NO content found:\n" + "\n".join(f"  • {f}" for f in missing_files)
        
        return f"""You are a document summarization assistant. Your task is to summarize EACH document individually.

{context_str}
{missing_note}

CRITICAL RULES:
1. Create a SEPARATE section for EACH document listed above.
2. Each section must start with: "## Document: [filename]"
3. For documents with content, summarize:
   - Main topics and key points
   - Important findings or data
   - Any conclusions or recommendations
4. For documents with NO content found, write exactly:
   "**Document: [filename]** - No relevant content found in this document."
5. Cite specific information using (Document Name, Page X, Line Y)
6. DO NOT merge or mix information across documents.
7. DO NOT add information not present in the context.

QUESTION: {query}

SUMMARY BY DOCUMENT:
"""

    # ─── PARTIAL MATCH PROMPT ──────────────────────────────────────────────
    elif partial_match:
        missing_note = ""
        if missing_files:
            missing_note = "\n\n⚠️ Documents with no relevant information:\n" + "\n".join(f"  • {f}" for f in missing_files)

        return f"""You are a document Q&A assistant. Provide the best answer using available information.

{context_str}
{missing_note}

RULES:
1. If information comes from multiple documents, cite each separately.
2. For documents with no information, state: "No relevant information found in [filename]."
3. Cite facts as (Document Name, Page X, Line Y).
4. Be honest about what you found and what you didn't.

QUESTION: {query}

ANSWER:"""

    # ─── SPECIFIC QUESTION PROMPT ──────────────────────────────────────────
    else:
        missing_note = ""
        if missing_files:
            missing_note = "\n\n⚠️ Documents with NO relevant information:\n" + "\n".join(f"  • {f}" for f in missing_files)

        return f"""You are a precise document Q&A assistant. Provide accurate answers using ONLY the context.

{context_str}
{missing_note}

RULES:
1. Answer the question directly and precisely.
2. Cite EVERY fact as (Document Name, Page X, Line Y).
3. If multiple documents have information, present it document-by-document.
4. If information is not found, say: "I couldn't find this in the available documents."
5. For documents with no relevant info, mention that explicitly.
6. DO NOT add information not present in the context.

QUESTION: {query}

ANSWER:"""


def build_no_context_response() -> str:
    return """I couldn't find relevant content to answer your question.

Suggestions:
• Try using more specific keywords
• Check if the document contains the information you need
• Try asking about a specific topic or section
• Rephrase your question with different terms"""