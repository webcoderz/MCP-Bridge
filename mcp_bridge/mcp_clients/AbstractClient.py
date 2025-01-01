import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional
from fastapi import HTTPException
from mcp import ClientSession, McpError
from mcp.types import CallToolResult, ListToolsResult, TextContent, ListResourcesResult, ListPromptsResult, GetPromptResult
from loguru import logger
from models.mcpServerStatus import McpServerStatus


class GenericMcpClient(ABC):
    name: str
    config: Any
    client: Any
    session: ClientSession

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    async def _maintain_session(self):
        pass

    async def _session_maintainer(self):
        while True:
            try:
                await self._maintain_session()
            except Exception as e:
                logger.trace(f"failed to maintain session for {self.name}: {e}")
                await asyncio.sleep(0.5)
    
    async def start(self):
        asyncio.create_task(self._session_maintainer())

    async def call_tool(
        self, name: str, arguments: dict, timeout: Optional[int] = None
    ) -> CallToolResult:
        await self._wait_for_session()

        try:
            async with asyncio.timeout(timeout):
                return await self.session.call_tool(
                    name=name,
                    arguments=arguments,
                )

        except asyncio.TimeoutError:
            logger.error(f"timed out calling tool: {name}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Timeout Error calling {name}")
                ],
                isError=True,
            )

        except McpError as e:
            logger.error(f"error calling {name}: {e}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error calling {name}: {e}")],
                isError=True,
            )
        
    async def get_prompt(self, prompt: str, arguments: dict[str, str]) -> GetPromptResult | None:
        await self._wait_for_session()

        try:
            return await self.session.get_prompt(prompt, arguments)
        except Exception as e:
            logger.error(f"error evaluating prompt: {e}")

        return None

    async def list_tools(self) -> ListToolsResult:
        # if session is None, then the client is not running
        # wait to see if it restarts
        await self._wait_for_session()

        try:
            return await self.session.list_tools()
        except Exception as e:
            logger.error(f"error listing tools: {e}")
            return ListToolsResult(tools=[])

    async def list_resources(self) -> ListResourcesResult:
        await self._wait_for_session()
        try: 
            return await self.session.list_resources()
        except Exception as e:
            logger.error(f"error listing resources: {e}")
            return ListResourcesResult(resources=[])
        
    async def list_prompts(self) -> ListPromptsResult:
        await self._wait_for_session()
        try: 
            return await self.session.list_prompts()
        except Exception as e:
            logger.error(f"error listing prompts: {e}")
            return ListPromptsResult(prompts=[])

    async def _wait_for_session(self, timeout: int = 10, http_error: bool = True):
        try:
            async with asyncio.timeout(timeout):
                while self.session is None:
                    await asyncio.sleep(1)

        except asyncio.TimeoutError:
            if http_error:
                raise HTTPException(status_code=500, detail="Could not connect to MCP server.")
            
            raise TimeoutError("Session initialization timed out.")
        
    async def status(self) -> McpServerStatus:
        """Get the status of the MCP server"""
        return McpServerStatus(
            name=self.name,
            online=self.session is not None,
            enabled=True
        )