"""Core data structures for text chunking."""

from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum


class ChunkType(Enum):
    """Types of chunks in the text chunking process."""
    FIRST = "first"  # First chunk
    MIDDLE = "middle"  # Middle chunks
    FINAL = "final"  # Last chunk in document


@dataclass
class TextChunk:
    """A chunk of text ready for LLM processing."""
    
    # Identity
    chunk_id: int
    chunk_type: ChunkType
    
    # Core Content
    text: str  # The chunk text
    token_count: int  # Number of tokens in this chunk
    
    # Metadata
    start_position: int  # Character position in original text
    end_position: int  # End character position in original text
    overlap_with_previous: int  # Characters overlapping with previous chunk
    boundary_info: Dict[str, Any]  # Boundary quality metrics
    
    # Quality Metrics
    boundary_score: float  # How good the boundary is (0-1)
    
    def __post_init__(self):
        """Validate chunk data after initialization."""
        if self.chunk_id < 0:
            raise ValueError("chunk_id must be non-negative")
        if not self.text.strip():
            raise ValueError("text cannot be empty")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if self.boundary_score < 0 or self.boundary_score > 1:
            raise ValueError("boundary_score must be between 0 and 1")


@dataclass
class ChunkingConfig:
    """Configuration for text chunking process."""
    
    max_chunk_tokens: int  # Maximum tokens per chunk
    overlap_tokens: int = 50  # Tokens to overlap between chunks
    tokenizer_name: str = "cl100k_base"  # Tokenizer to use
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if self.overlap_tokens >= self.max_chunk_tokens:
            raise ValueError("overlap_tokens must be less than max_chunk_tokens")


