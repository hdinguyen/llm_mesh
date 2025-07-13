"""Text Chunker - A simple system for chunking text for LLM preprocessing."""

from .models import ChunkType, TextChunk, ChunkingConfig
from .text_chunker import TextChunker
from .simple_token_manager import TokenManager
from .simple_boundary_optimizer import BoundaryOptimizer

__all__ = [
    "ChunkType",
    "TextChunk", 
    "ChunkingConfig",
    "TextChunker",
    "TokenManager",
    "BoundaryOptimizer",
]

__version__ = "0.1.0"