import asyncio
import re
import subprocess
import time
from typing import List, Tuple, Optional
from pathlib import Path

from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.types import Tool
from mcp.client.stdio import stdio_client

from loguru import logger

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    def get_tools_information(self, tools:List[Tool]):
        tools_information = []
        for tool in tools:
            tools_information.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
                "annotations": tool.annotations
            })
        return str(tools_information)

    async def connect(self, mcp_config:dict):
        server_param = StdioServerParameters(
            command=mcp_config['command'],
            args=mcp_config['args'],
            env=None if 'env' not in mcp_config.keys() else mcp_config['env']
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_param))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        try:
            await self.session.initialize()
            response = await self.session.list_tools()
            tools_information = self.get_tools_information(response.tools)
            return tools_information
        except Exception as e:
            logger.error(f"Error during setup MCP: {e}")
            raise e

    async def disconnect(self):
        """Properly cleanup all async contexts"""
        try:
            await self.exit_stack.aclose()
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
        finally:
            self.session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

def parse_chapter_text(raw_text: str):
    # Split into lines for easier processing
    lines = raw_text.split('\n')
    title = None
    content_lines = []
    in_content = False

    for line in lines:
        if line.startswith('Title:'):
            # Remove 'Title:' and strip whitespace
            title_line = line[len('Title:'):].strip()
            # Remove trailing '– Băng Phách' or similar
            title = re.sub(r'\s*–\s*Băng Phách$', '', title_line)
        elif line.startswith('Content:'):
            in_content = True
            continue  # Skip the 'Content:' line itself
        elif in_content:
            # Skip unwanted headers or repeated titles
            if line.startswith('###') or 'Vương Tử Ngược Bắc Em Xuôi Nam' in line:
                continue
            if line.strip() == '':
                continue  # Skip empty lines
            content_lines.append(line.strip())

    content = '\n'.join(content_lines).strip()
    return {'title': title, 'content': content}

async def main():
    """Main entry point."""
    print("Vietnamese Novel Chapter Fetcher with MCP")
    print("=" * 45)

    # Test MCPClient
    mcp_client = MCPClient()
    await mcp_client.connect({
      "command": "npx",
      "args": ["-y", "fetcher-mcp"]
    })

    urls = []

    # Read URLs and create concurrent fetch tasks
    fetch_tasks = []
    with open("/Users/nguyenh/workspace/trans/agent_mesh/chunking/vuong_tu_nguoc_bac_em_xuoi_nam.txt", "r") as f:
        for line in f:
            url = line.strip()
            if url:  # Skip empty lines
                task = mcp_client.session.call_tool(
                    "fetch_url",
                    {
                        "url": url,
                        "extractContent": True
                    }
                )
                fetch_tasks.append(task)
    
    # Wait for all futures and collect responses in order
    responses = await asyncio.gather(*fetch_tasks, return_exceptions=True)
    
    with open("/Users/nguyenh/workspace/trans/agent_mesh/chunking/vuong_tu_nguoc_bac_em_xuoi_nam_raw.txt", "w") as f:
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Failed to fetch URL {i}: {response}")
            else:
                try:
                    parsed_chapter = parse_chapter_text(response.content[0].text)
                    f.write(f"Title: {parsed_chapter['title']}\nContent: {parsed_chapter['content']}\n\n-----\n")
                except Exception as e:
                    logger.error(f"Failed to parse chapter {i+1}: {e}")

    await mcp_client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
