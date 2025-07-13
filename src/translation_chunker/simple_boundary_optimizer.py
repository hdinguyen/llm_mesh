"""Simple boundary optimization for text chunking."""

import re
from typing import Tuple, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .simple_token_manager import TokenManager


class BoundaryOptimizer:
    """Simple boundary optimization for clean text chunks."""
    
    def __init__(self, token_manager: TokenManager):
        """Initialize boundary optimizer."""
        self.token_manager = token_manager
        
        # Configure LangChain splitter for semantic boundaries
        self.langchain_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings
                "! ",    # Exclamation endings
                "? ",    # Question endings
                "; ",    # Semicolon breaks
                ", ",    # Comma breaks
                " ",     # Word breaks
            ],
            chunk_size=1000,  # Will be updated dynamically
            chunk_overlap=100,
            length_function=self.token_manager.count_tokens,
            is_separator_regex=False,
        )
    
    def find_optimal_boundary(self, text: str, target_tokens: int) -> Tuple[int, float]:
        """Find the best boundary position for target token count."""
        if not text:
            return 0, 0.0
        
        total_tokens = self.token_manager.count_tokens(text)
        if total_tokens <= target_tokens:
            return len(text), 1.0
        
        # Get approximate position based on tokens
        approx_pos = self.token_manager.find_token_boundary(text, target_tokens)
        
        # Look for better semantic boundaries nearby
        search_window = min(500, len(text) // 10)  # Search within 500 chars or 10% of text
        start_pos = max(0, approx_pos - search_window)
        end_pos = min(len(text), approx_pos + search_window)
        
        best_pos = approx_pos
        best_score = 0.5  # Base score
        
        # Check for paragraph breaks
        for match in re.finditer(r'\n\n+', text[start_pos:end_pos]):
            pos = start_pos + match.end()
            if self._is_within_token_limit(text[:pos], target_tokens):
                score = 1.0  # Paragraph breaks are best
                if score > best_score:
                    best_pos = pos
                    best_score = score
        
        # Check for sentence endings if no good paragraph breaks
        if best_score < 0.9:
            for match in re.finditer(r'[.!?]\s+', text[start_pos:end_pos]):
                pos = start_pos + match.end()
                if self._is_within_token_limit(text[:pos], target_tokens):
                    score = 0.8  # Sentence breaks are good
                    if score > best_score:
                        best_pos = pos
                        best_score = score
        
        # Check for line breaks if no good sentence breaks
        if best_score < 0.7:
            for match in re.finditer(r'\n', text[start_pos:end_pos]):
                pos = start_pos + match.end()
                if self._is_within_token_limit(text[:pos], target_tokens):
                    score = 0.6  # Line breaks are okay
                    if score > best_score:
                        best_pos = pos
                        best_score = score
        
        return best_pos, best_score
    
    def _is_within_token_limit(self, text: str, target_tokens: int, tolerance: int = 50) -> bool:
        """Check if text is within token limit with tolerance."""
        tokens = self.token_manager.count_tokens(text)
        return tokens <= target_tokens + tolerance
    
    def create_overlap(self, text: str, overlap_tokens: int) -> str:
        """Create overlap text from the end of previous chunk."""
        if not text or overlap_tokens <= 0:
            return ""
        
        # Find approximate character position for overlap
        overlap_pos = self.token_manager.find_token_boundary(text, overlap_tokens)
        
        if overlap_pos >= len(text):
            return text
        
        # Try to break at sentence boundary within overlap region
        overlap_text = text[-overlap_pos:] if overlap_pos > 0 else ""
        
        # Look for sentence start in overlap
        sentences = re.split(r'[.!?]\s+', overlap_text)
        if len(sentences) > 1:
            # Start from complete sentence
            return sentences[-1] if sentences[-1] else overlap_text
        
        return overlap_text
    
    def get_boundary_info(self, text: str, position: int) -> Dict[str, Any]:
        """Get information about boundary quality."""
        if position <= 0 or position >= len(text):
            return {"position": position, "type": "edge", "score": 0.0}
        
        # Check what type of boundary this is
        before = text[max(0, position-5):position]
        after = text[position:min(len(text), position+5)]
        context = before + after
        
        boundary_type = "word"
        score = 0.3
        
        if '\n\n' in context:
            boundary_type = "paragraph"
            score = 1.0
        elif re.search(r'[.!?]\s', context):
            boundary_type = "sentence"
            score = 0.8
        elif '\n' in context:
            boundary_type = "line"
            score = 0.6
        elif ' ' in context:
            boundary_type = "word"
            score = 0.4
        else:
            boundary_type = "character"
            score = 0.1
        
        return {
            "position": position,
            "type": boundary_type,
            "score": score,
            "before_context": before,
            "after_context": after
        }
    
    def split_with_langchain(self, text: str, target_tokens: int) -> list[str]:
        """Use LangChain for splitting when needed."""
        # Update chunk size
        self.langchain_splitter._chunk_size = target_tokens
        self.langchain_splitter._chunk_overlap = min(100, target_tokens // 10)
        
        return self.langchain_splitter.split_text(text)