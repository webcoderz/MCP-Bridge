from lmos_openai_types import (
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    ChatCompletionRequestMessage,
)

from .utils import chat_completion_add_tools
from .genericHttpxClient import client
from mcp_clients.McpClientManager import ClientManager
from tool_mappers import mcp2openai
from loguru import logger
import json


async def chat_completions(
    request: CreateChatCompletionRequest,
) -> CreateChatCompletionResponse:
    """performs a chat completion using the inference server"""

    request = await chat_completion_add_tools(request)

    while True:
        # logger.debug(request.model_dump_json())

        text = (
            await client.post(
                "/chat/completions",
                json=request.model_dump(
                    exclude_defaults=True, exclude_none=True, exclude_unset=True
                ),
            )
        ).text
        logger.debug(text)
        try:
            response = CreateChatCompletionResponse.model_validate_json(text)
        except Exception:
            return

        msg = response.choices[0].message
        msg = ChatCompletionRequestMessage(
            role="assistant",
            content=msg.content,
            tool_calls=msg.tool_calls,
        )  # type: ignore
        request.messages.append(msg)

        logger.debug(f"finish reason: {response.choices[0].finish_reason}")
        if response.choices[0].finish_reason.value in ["stop", "length"]:
            logger.debug("no tool calls found")
            return response

        logger.debug("tool calls found")
        for tool_call in response.choices[0].message.tool_calls.root:
            logger.debug(
                f"tool call: {tool_call.function.name} arguments: {json.loads(tool_call.function.arguments)}"
            )

            # FIXME: this can probably be done in parallel using asyncio gather
            session = await ClientManager.get_client_from_tool(tool_call.function.name)
            tool_call_result = await session.call_tool(
                name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments),
            )

            logger.debug(
                f"tool call result for {tool_call.function.name}: {tool_call_result.model_dump()}"
            )

            logger.debug(f"tool call result content: {tool_call_result.content}")

            tools_content = [{"type": "text", "text": part.text} for part in tool_call_result.content]
            request.messages.append(
                ChatCompletionRequestMessage.model_validate(
                    {
                        "role": "tool",
                        "content": tools_content,
                        "tool_call_id": tool_call.id,
                    }
                )
            )


            logger.debug("sending next iteration of chat completion request")
