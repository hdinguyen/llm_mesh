"""Simple token management for text chunking."""

import tiktoken
from .models import ChunkingConfig


class TokenManager:
    """Simple token counting and management for text chunks."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize token manager with configuration."""
        self.config = config
        self.tokenizer = tiktoken.get_encoding(config.tokenizer_name)
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the configured tokenizer."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))
    
    def find_token_boundary(self, text: str, target_tokens: int) -> int:
        """Find character position that gives approximately target_tokens."""
        if not text:
            return 0
            
        total_tokens = self.count_tokens(text)
        if total_tokens <= target_tokens:
            return len(text)
        
        # Binary search for the right character position
        left, right = 0, len(text)
        best_pos = 0
        
        while left <= right:
            mid = (left + right) // 2
            chunk_tokens = self.count_tokens(text[:mid])
            
            if chunk_tokens <= target_tokens:
                best_pos = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return best_pos
    
    def calculate_overlap_tokens(self, text: str, overlap_chars: int) -> int:
        """Calculate tokens in overlap portion."""
        if overlap_chars <= 0 or overlap_chars >= len(text):
            return 0
        return self.count_tokens(text[-overlap_chars:])
    
    def get_effective_chunk_size(self) -> int:
        """Get effective chunk size accounting for overlap."""
        return self.config.max_chunk_tokens - self.config.overlap_tokens