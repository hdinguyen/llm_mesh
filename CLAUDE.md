# Translation Text Chunker - Requirements & Design

## Problem Statement

Build a Python system to intelligently chunk very long text for translation while maintaining quality and respecting token limits. The system must optimize for translation consistency across chunks while minimizing API calls.

## Core Requirements

### 1. Dynamic Model Configuration
- Accept `max_context_tokens` and `max_output_tokens` as parameters
- Example: max_context=1.05M, max_output=66K, but should work with any model limits
- Support different LLM models with varying token limits

### 2. Token Management Strategy
- **Target**: Input ~60K tokens → Expected output ~60K tokens (1:1 ratio for translation)
- **Rule**: Never exceed max_output_token limit
- **Optimization**: Get as close as possible to the limit without exceeding
- **Context Inclusion**: Token count must include ALL components (text + context + memory)

### 3. Context Management Strategy
- **Chunk 1**: Pure text (no previous context) - uses full available tokens
- **Chunk 2+**: Text + previous chunk summary + extracted memory
- **Formula**: `current_text_tokens + context_summary_tokens + memory_tokens ≤ max_output_tokens`

## Design Decisions

### Memory Management
- **Scope**: Accumulate from ALL previous chunks for better context
- **Sliding Window**: Keep last 3-5 chunks in detail + condensed summary of older chunks
- **Context Cap**: Total context limited to ~15-20% of max_output_tokens
- **Prevention**: Avoid token bloat while maintaining recent context

### Text Overlap Strategy
- **Semantic Overlap**: Include the last complete paragraph/section from previous chunk
- **Contextual Bridge**: Add 1-2 sentences before and after main chunk boundaries
- **Purpose**: Ensure no narrative breaks and provide translation continuity

### Three-Part Input Structure
For every translation chunk (except the first):
1. **General Context**: Accumulated important info (names, terms, domain vocabulary)
2. **Previous Chunk Context**: Summary of immediate previous chunk's main themes
3. **Raw Chunk Text**: Current content with smart overlaps

### Context Prioritization
- **General Context**: Most frequent/important terms, names, domain-specific vocabulary
- **Previous Chunk Context**: Recent chunk summary and translation patterns
- **Raw Chunk Text**: Current content with intelligent boundary selection

### Text Boundary Preferences
- **Priority Order**: Paragraph boundaries > sentence boundaries > word boundaries
- **Semantic Markers**: Use headings, bullet points, section breaks when available
- **Avoid Breaking**: Middle of quotes, lists, technical descriptions, or dialogue

## Hybrid Architecture Approach

### Recommended Technology Stack
Based on research analysis, we'll use a **hybrid approach** combining the best of existing packages with custom translation-specific implementations:

#### Foundation Libraries
1. **LangChain Text Splitters** - Primary chunking engine
   - `RecursiveCharacterTextSplitter` for semantic boundary respect
   - Built-in tiktoken integration for accurate token counting
   - Mature production-ready ecosystem
   
2. **semchunk** - Enhanced semantic chunking
   - 85% faster performance than alternatives
   - Superior semantic boundary detection
   - Production-proven in legal document processing

#### Custom Implementation Layers
3. **Translation Context Manager** (Custom)
   - Three-part input structure management
   - Cross-chunk memory accumulation
   - Translation-specific context extraction

4. **Token Optimization Engine** (Custom)
   - Dynamic token allocation across context components
   - Sliding window memory management
   - Boundary optimization for translation quality

### Technical Specifications

#### Output Format
```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class ChunkType(Enum):
    FIRST = "first"  # No previous context
    SUBSEQUENT = "subsequent"  # Has previous context
    FINAL = "final"  # Last chunk in document

@dataclass
class TranslationChunk:
    # Identity
    chunk_id: int
    chunk_type: ChunkType
    
    # Core Content
    raw_text: str  # Pure text for this chunk
    general_context: str  # Accumulated important info (names, terms, etc.)
    previous_chunk_context: str  # Recent chunk summary and patterns
    full_input_text: str  # Combined text ready for translation API
    
    # Metadata
    token_breakdown: Dict[str, int]  # Tokens per component
    overlap_info: Dict[str, any]  # Overlap with adjacent chunks
    boundary_info: Dict[str, any]  # Boundary quality metrics
    extraction_metadata: Dict[str, List[str]]  # Extracted entities/terms
    
    # Quality Metrics
    semantic_coherence_score: float  # Boundary quality score
    context_completeness_score: float  # Context adequacy score
```

#### Context Extraction Requirements
- **Proper Nouns**: Names, locations, organizations, brands
- **Technical Vocabulary**: Domain-specific terms, acronyms, jargon
- **Numerical Data**: Measurements, dates, quantities, percentages
- **Formatting Preservation**: Quotes, emphasis, special characters
- **Translation Patterns**: Consistent terminology tracking across chunks
- **Semantic Relationships**: Subject-object relationships, references

#### Token Management Architecture
- **Primary Tokenizer**: tiktoken for OpenAI models (configurable)
- **Fallback Support**: Hugging Face tokenizers, custom counters
- **Real-time Tracking**: Dynamic token allocation per component
- **Memory Management**: Sliding window for context accumulation
- **Optimization**: Dynamic context sizing based on available tokens

## Quality Optimization Features

### Chunking Quality
1. **Natural Boundaries**: Respect sentence and paragraph structure
2. **Context Preservation**: Maintain narrative flow and meaning
3. **Terminology Consistency**: Track and maintain consistent translations
4. **Overlap Management**: Smart overlap to prevent context loss

### Translation Quality
1. **Context Continuity**: Ensure consistent terminology across chunks
2. **Narrative Flow**: Preserve story/document structure
3. **Domain Awareness**: Adapt to technical, legal, literary content
4. **Reference Maintenance**: Keep track of names, places, concepts

## Implementation Strategy

### Phase 1: Foundation Setup
1. **Environment Configuration**
   - Install LangChain: `pip install langchain-text-splitters tiktoken`
   - Install semchunk: `pip install semchunk`
   - Setup tokenizer configurations for multiple models

2. **Core Architecture**
   - `TranslationChunker` class as main orchestrator
   - `ContextExtractor` for translation-specific information extraction
   - `TokenManager` for dynamic token allocation and tracking
   - `BoundaryOptimizer` for semantic boundary detection

### Phase 2: Custom Components
3. **Context Management System**
   - Sliding window memory for previous chunks
   - General context accumulation and condensation
   - Translation pattern tracking across chunks

4. **Token Optimization Engine**
   - Dynamic allocation: 15-20% for context, 80-85% for content
   - Real-time token counting with multiple tokenizer support
   - Boundary adjustment based on available token space

### Phase 3: Quality Optimization
5. **Semantic Boundary Detection**
   - Hybrid approach: LangChain + semchunk for optimal boundaries
   - Translation-aware overlap management
   - Context bridge creation between chunks

6. **Translation Quality Features**
   - Entity consistency tracking
   - Terminology database maintenance
   - Cross-chunk reference resolution

### Implementation Priorities
1. **Translation Quality**: Primary focus on maintaining translation consistency
2. **Token Efficiency**: Maximize content while respecting limits  
3. **Hybrid Performance**: Leverage best of both LangChain and semchunk
4. **Context Intelligence**: Smart extraction and management of relevant information
5. **Boundary Optimization**: Natural, semantic chunk boundaries
6. **Scalability**: Handle very long documents efficiently

### IDE Compatibility Notes
- **Claude-Compatible**: Use dataclasses, type hints, clear documentation
- **Cursor-Optimized**: Modular architecture for AI-assisted development
- **Clear Interfaces**: Well-defined class boundaries for code completion
- **Comprehensive Testing**: Unit tests for each component

## Success Criteria

- Chunks never exceed token limits
- Translation quality maintained across all chunks
- Context information preserved and utilized effectively
- Minimal API calls while maximizing translation accuracy
- Robust handling of different text types and structures

## Package Dependencies

### Required Libraries
```bash
# Core chunking and token management
pip install langchain-text-splitters tiktoken semchunk

# NLP and language processing  
pip install spacy nltk
python -m spacy download en_core_web_sm

# Data handling and utilities
pip install dataclasses-json pydantic typing-extensions

# Optional: Enhanced tokenizer support
pip install transformers tokenizers

# Development and testing
pip install pytest pytest-cov black isort mypy
```