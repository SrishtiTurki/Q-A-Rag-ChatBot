from typing import List


def chunk_pages(
    pages: List[dict],
    chunk_size: int = 500,
    overlap: int = 50,
    source: str = "unknown"
) -> List[dict]:
    """
    Takes page-aware output from file_parser and chunks each page,
    tracking exact page number and estimated line/paragraph number per chunk.
    """
    if not pages:
        return []

    chunks = []
    chunk_index = 0

    for page in pages:
        text = page["text"].strip()
        page_number = page["page_number"]

        if not text:
            continue

        words = text.split()
        page_lines = text.splitlines()

        if not words:
            continue

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_str = " ".join(chunk_words)

            # Estimate line and paragraph info
            line_start, para_start = _estimate_position(text, page_lines, chunk_words[:10])
            
            # Count total lines in chunk
            chunk_lines = chunk_str.count('\n') + 1 if chunk_str else 0

            chunks.append({
                "text": chunk_str,
                "source": source,
                "chunk_index": chunk_index,
                "page_number": page_number,
                "line_start": line_start,
                "line_end": line_start + chunk_lines - 1 if chunk_lines > 0 else line_start,
                "paragraph_start": para_start,
                "uid": f"{source}__{chunk_index}"  # unique across all files
            })

            chunk_index += 1
            start += chunk_size - overlap

    return chunks


def _estimate_position(full_text: str, lines: list, first_words: list) -> tuple:
    """
    Estimates the line and paragraph number where a chunk starts within a page.
    Returns (line_number, paragraph_number)
    """
    if not first_words or not lines:
        return (1, 1)

    search_phrase = " ".join(first_words)
    char_pos = full_text.find(search_phrase)

    if char_pos == -1:
        return (1, 1)

    # Count newlines before this position
    text_before = full_text[:char_pos]
    line_number = text_before.count("\n") + 1
    
    # Count paragraph breaks (double newlines) before this position
    # Paragraphs are separated by blank lines (two newlines)
    para_text = text_before
    # Remove trailing newlines for accurate paragraph counting
    para_text = para_text.rstrip('\n')
    if para_text:
        # Count blank lines as paragraph separators
        para_count = 1  # Start at paragraph 1
        for i, char in enumerate(para_text):
            if char == '\n' and i + 1 < len(para_text) and para_text[i+1] == '\n':
                para_count += 1
                # Skip the second newline
                continue
        paragraph_number = para_count
    else:
        paragraph_number = 1

    return (line_number, paragraph_number)


def chunk_multiple_files(file_pages: List[dict]) -> List[dict]:
    """
    Convenience wrapper for multiple uploaded files.
    """
    all_chunks = []
    for file in file_pages:
        chunks = chunk_pages(
            pages=file["pages"],
            source=file["source"]
        )
        all_chunks.extend(chunks)
    return all_chunks