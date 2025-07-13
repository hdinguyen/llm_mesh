"""Simple text chunker for LLM preprocessing."""

from typing import List, Iterator, Dict, Any
from .models import ChunkingConfig, TextChunk, ChunkType
from .simple_token_manager import TokenManager
from .simple_boundary_optimizer import BoundaryOptimizer


class TextChunker:
    """Simple text chunker for preparing LLM input."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize the text chunker with configuration."""
        self.config = config
        self.token_manager = TokenManager(config)
        self.boundary_optimizer = BoundaryOptimizer(self.token_manager)
        
    def chunk_text(self, text: str) -> List[TextChunk]:
        """Chunk text into token-limited chunks with good boundaries."""
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        
        chunks = []
        remaining_text = text
        chunk_id = 0
        current_position = 0
        
        while remaining_text.strip():
            # Calculate target tokens for this chunk
            target_tokens = self.token_manager.get_effective_chunk_size()
            
            # Find optimal boundary
            boundary_pos, boundary_score = self.boundary_optimizer.find_optimal_boundary(
                remaining_text, target_tokens
            )
            
            # Extract chunk text
            chunk_text = remaining_text[:boundary_pos].strip()
            if not chunk_text:
                break
            
            # Handle overlap for non-first chunks
            overlap_chars = 0
            if chunk_id > 0 and self.config.overlap_tokens > 0:
                overlap_text = self.boundary_optimizer.create_overlap(
                    chunk_text, self.config.overlap_tokens
                )
                overlap_chars = len(overlap_text)
            
            # Determine chunk type
            remaining_after = remaining_text[boundary_pos:].strip()
            if chunk_id == 0:
                chunk_type = ChunkType.FIRST
            elif not remaining_after:
                chunk_type = ChunkType.FINAL
            else:
                chunk_type = ChunkType.MIDDLE
            
            # Create chunk
            chunk = TextChunk(
                chunk_id=chunk_id,
                chunk_type=chunk_type,
                text=chunk_text,
                token_count=self.token_manager.count_tokens(chunk_text),
                start_position=current_position,
                end_position=current_position + len(chunk_text),
                overlap_with_previous=overlap_chars,
                boundary_info=self.boundary_optimizer.get_boundary_info(remaining_text, boundary_pos),
                boundary_score=boundary_score
            )
            
            chunks.append(chunk)
            
            # Move to next chunk
            remaining_text = remaining_text[boundary_pos:].strip()
            current_position += boundary_pos
            chunk_id += 1
            
            # Safety check
            if boundary_pos == 0:
                break
        
        return chunks
    
    def chunk_text_iterator(self, text: str) -> Iterator[TextChunk]:
        """Yield chunks one at a time for memory efficiency."""
        for chunk in self.chunk_text(text):
            yield chunk
    
    def get_chunking_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about the chunking process."""
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        total_tokens = sum(chunk.token_count for chunk in chunks)
        avg_boundary_score = sum(chunk.boundary_score for chunk in chunks) / total_chunks
        
        boundary_types = {}
        for chunk in chunks:
            boundary_type = chunk.boundary_info.get("type", "unknown")
            boundary_types[boundary_type] = boundary_types.get(boundary_type, 0) + 1
        
        return {
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": total_tokens / total_chunks,
            "avg_boundary_score": avg_boundary_score,
            "boundary_type_distribution": boundary_types,
            "max_chunk_tokens": max(chunk.token_count for chunk in chunks),
            "min_chunk_tokens": min(chunk.token_count for chunk in chunks),
            "total_overlap_chars": sum(chunk.overlap_with_previous for chunk in chunks)
        }
    
    def validate_chunks(self, chunks: List[TextChunk]) -> List[str]:
        """Validate that chunks meet requirements."""
        issues = []
        
        for i, chunk in enumerate(chunks):
            # Check token limits
            if chunk.token_count > self.config.max_chunk_tokens:
                issues.append(f"Chunk {i} exceeds token limit: {chunk.token_count} > {self.config.max_chunk_tokens}")
            
            # Check chunk type consistency
            if i == 0 and chunk.chunk_type != ChunkType.FIRST:
                issues.append(f"First chunk should be ChunkType.FIRST, got {chunk.chunk_type}")
            
            if i == len(chunks) - 1 and chunk.chunk_type != ChunkType.FINAL and len(chunks) > 1:
                issues.append(f"Last chunk should be ChunkType.FINAL, got {chunk.chunk_type}")
            
            # Check for empty chunks
            if not chunk.text.strip():
                issues.append(f"Chunk {i} has empty text")
        
        return issues
    
    def combine_chunks(self, chunks: List[TextChunk]) -> str:
        """Combine chunks back into original text (for testing)."""
        if not chunks:
            return ""
        
        # Sort by chunk_id to ensure correct order
        sorted_chunks = sorted(chunks, key=lambda c: c.chunk_id)
        
        # For simple reconstruction, just join texts
        # Note: This won't perfectly recreate original due to overlaps
        return " ".join(chunk.text for chunk in sorted_chunks)
    
    def chunk_to_dict(self, chunk: TextChunk) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "chunk_id": chunk.chunk_id,
            "chunk_type": chunk.chunk_type.value,
            "text": chunk.text,
            "token_count": chunk.token_count,
            "start_position": chunk.start_position,
            "end_position": chunk.end_position,
            "overlap_with_previous": chunk.overlap_with_previous,
            "boundary_info": chunk.boundary_info,
            "boundary_score": chunk.boundary_score
        }