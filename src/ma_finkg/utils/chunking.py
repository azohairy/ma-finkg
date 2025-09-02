def chunk_large_text(text: str, max_size: int = 3000) -> list:
    """Split large text into manageable chunks for LLM processing."""
    if len(text) <= max_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_size
        if end >= len(text):
            chunks.append(text[start:])
            break
            
        # Find sentence boundary within last 200 chars
        boundary = text.rfind('.', end-200, end)
        if boundary > start:
            end = boundary + 1
        
        chunks.append(text[start:end])
        start = end
    
    return chunks