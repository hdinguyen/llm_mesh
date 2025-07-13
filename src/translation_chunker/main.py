"""Example usage of the text chunker."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from translation_chunker import TextChunker, ChunkingConfig

from collections.abc import AsyncGenerator

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server

server = Server()

@server.agent(
    name="chunk_text",
    description="Chunk text into smaller chunks",
)
async def chunk_text(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    # Configure for chunking with token limits
    config = ChunkingConfig(
        max_chunk_tokens=10000,      # Max tokens per chunk
        overlap_tokens=200,         # Overlap between chunks
        tokenizer_name="cl100k_base"  # OpenAI tokenizer
    )
    
    # Initialize chunker
    chunker = TextChunker(config)
    
    # Example long text (replace with actual long text)
    sample_text = ""
    with open("/Users/nguyenh/workspace/trans/agent_mesh/chunking/story.txt", "r") as f:
        sample_text = f.read()
    
    try:
        # Chunk the text
        chunks = chunker.chunk_text(sample_text)
        
        # Prepare all chunk parts for a single message
        for chunk in chunks:
            yield MessagePart(
                content_type="text/plain",
                content=chunk.text
            )
        

        # Get statistics
        stats = chunker.get_chunking_stats(chunks)
        print("Chunking Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Validate chunks
        issues = chunker.validate_chunks(chunks)
        if issues:
            print("\nValidation Issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\nAll chunks passed validation!")
            
    except Exception as e:
        print(f"Error: {e}")

server.run()