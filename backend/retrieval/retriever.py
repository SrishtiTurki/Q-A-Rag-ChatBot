import numpy as np
from typing import List, Optional
from retrieval.vector_store import search

SIMILARITY_THRESHOLD = 0.15
PARTIAL_MATCH_THRESHOLD = 0.07

def retrieve(
    query_embedding: np.ndarray,
    query_text: str = "",
    top_k: int = 15,
    threshold: float = SIMILARITY_THRESHOLD,
    filter_sources: Optional[List[str]] = None,
    include_all_sources: bool = False
) -> dict:
    """Retrieves the most relevant chunks with fair source distribution."""
    
    # For summarization, get more chunks from ALL documents
    if include_all_sources:
        top_k = 150  # Get many more chunks
        threshold = 0.03  # Lower threshold to include more content
    
    raw_results = search(query_embedding, top_k=top_k)

    if not raw_results:
        return {"found": False, "partial": False, "chunks": [], "sources": [], "pdf_sources": {}}

    # Deduplicate and filter
    seen = {}
    for score, metadata in raw_results:
        # Apply source filter if specified
        if filter_sources is not None and len(filter_sources) > 0:
            clean_source = metadata['source'].replace(' [images]', '')
            if clean_source not in filter_sources and metadata['source'] not in filter_sources:
                continue
        
        uid = metadata.get("uid", f"{metadata['source']}_{metadata['chunk_index']}")
        if uid not in seen or score > seen[uid][0]:
            seen[uid] = (score, metadata)

    if not seen:
        return {"found": False, "partial": False, "chunks": [], "sources": [], "pdf_sources": {}}

    # Separate matches by threshold
    full_matches = [
        {**meta, "score": score}
        for uid, (score, meta) in seen.items()
        if score >= threshold
    ]

    partial_matches = [
        {**meta, "score": score}
        for uid, (score, meta) in seen.items()
        if PARTIAL_MATCH_THRESHOLD <= score < threshold
    ]

    def group_by_source(matches):
        source_groups = {}
        for match in matches:
            clean_source = match['source'].replace(' [images]', '')
            if clean_source not in source_groups:
                source_groups[clean_source] = []
            source_groups[clean_source].append(match)
        return source_groups

    def balance_results(matches, max_per_source=5):
        """Get balanced results from all sources."""
        source_groups = group_by_source(matches)
        
        # Sort each source's matches by score
        for source in source_groups:
            source_groups[source].sort(key=lambda x: x["score"], reverse=True)
        
        # Take top N from each source
        balanced = []
        for source, source_matches in source_groups.items():
            # For summarization, take more from each source
            max_ps = max_per_source * 3 if include_all_sources else max_per_source
            balanced.extend(source_matches[:max_ps])
        
        # Sort by score globally
        balanced.sort(key=lambda x: x["score"], reverse=True)
        return balanced

    def build_citation_info(matches):
        """Build detailed source information for citations."""
        pdf_sources = {}
        for match in matches:
            clean_source = match['source'].replace(' [images]', '')
            is_image = '[images]' in match['source']
            
            if clean_source not in pdf_sources:
                pdf_sources[clean_source] = {
                    'scores': [],
                    'pages': set(),
                    'chunks': [],
                    'image_chunks': []
                }
            
            chunk_info = {
                'text': match['text'],
                'page': match.get('page_number', '?'),
                'line_start': match.get('line_start', '?'),
                'line_end': match.get('line_end', '?'),
                'score': match['score'],
                'is_image': is_image
            }
            
            if is_image:
                pdf_sources[clean_source]['image_chunks'].append(chunk_info)
            else:
                pdf_sources[clean_source]['chunks'].append(chunk_info)
                
            pdf_sources[clean_source]['scores'].append(match['score'])
            pdf_sources[clean_source]['pages'].add(match.get('page_number', '?'))
        
        # Convert pages to sorted list
        for source in pdf_sources:
            pdf_sources[source]['pages'] = sorted(list(pdf_sources[source]['pages']))
            
        return pdf_sources

    if full_matches:
        max_per = 10 if include_all_sources else 4
        balanced_results = balance_results(full_matches, max_per_source=max_per)
        pdf_sources = build_citation_info(full_matches)
        sources = list(pdf_sources.keys())
        
        return {
            "found": True,
            "partial": False,
            "chunks": balanced_results,
            "sources": sources,
            "pdf_sources": pdf_sources,
            "all_sources": sources
        }

    elif partial_matches:
        max_per = 6 if include_all_sources else 3
        balanced_results = balance_results(partial_matches, max_per_source=max_per)
        pdf_sources = build_citation_info(partial_matches)
        sources = list(pdf_sources.keys())
        
        return {
            "found": True,
            "partial": True,
            "chunks": balanced_results,
            "sources": sources,
            "pdf_sources": pdf_sources,
            "all_sources": sources
        }

    return {"found": False, "partial": False, "chunks": [], "sources": [], "pdf_sources": {}, "all_sources": []}