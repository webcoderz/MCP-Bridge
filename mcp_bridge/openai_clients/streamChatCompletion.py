import json
from typing import Optional
from lmos_openai_types import (
    ChatCompletionMessageToolCall,
    ChatCompletionRequestMessage,
    CreateChatCompletionRequest,
    CreateChatCompletionStreamResponse,
    Function1,
)
from .utils import chat_completion_add_tools
from models import SSEData
from .genericHttpxClient import client
from mcp_clients.McpClientManager import ClientManager
from tool_mappers import mcp2openai
from loguru import logger
from httpx_sse import aconnect_sse

from sse_starlette.sse import EventSourceResponse, ServerSentEvent
import asyncio
from starlette.websockets import WebSocketDisconnect
from starlette.responses import StreamingResponse


async def streaming_chat_completions(request: CreateChatCompletionRequest):
    try:
        return EventSourceResponse(
            content=chat_completions(request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    except Exception as e:
        logger.error(f"Exception in streaming_chat_completions: {e}")
        # You may return a 500 response or handle it differently if needed.
        # For example:
        # return JSONResponse({"error": str(e)}, status_code=500)


async def chat_completions(request: CreateChatCompletionRequest):
    """performs a chat completion using the inference server with SSE and tool calls"""
    request.stream = True

    request = await chat_completion_add_tools(request)

    fully_done = False
    try:
        while not fully_done:
            json_data = request.model_dump_json(
                exclude_defaults=True, exclude_none=True, exclude_unset=True
            )

            logger.debug(f"Request JSON: {json_data}")

            last: Optional[CreateChatCompletionStreamResponse] = None
            tool_call_name: str = ""
            tool_call_json: str = ""
            should_forward: bool = True
            response_content: str = ""
            tool_call_id: str = ""

            # Connect to the SSE endpoint
            async with aconnect_sse(
                client, "post", "/chat/completions", content=json_data
            ) as event_source:
                async for sse in event_source.aiter_sse():
                    event = sse.event
                    data = sse.data
                    id = sse.id
                    retry = sse.retry

                    logger.debug(
                        f"SSE received - event: {event}, data: {data}, id: {id}, retry: {retry}"
                    )

                    if data == "[DONE]":
                        # Inference server signaled end of stream
                        logger.debug("[DONE] received from server")
                        break

                    parsed_data = CreateChatCompletionStreamResponse.model_validate_json(
                        data
                    )

                    # Extract content delta
                    content = parsed_data.choices[0].delta.content
                    content = content if content is not None else ""
                    response_content += content

                    finish_reason = parsed_data.choices[0].finish_reason
                    if finish_reason is not None:
                        logger.debug(f"Finish reason: {finish_reason.value}")
                        if finish_reason.value in ["stop", "length"]:
                            fully_done = True
                            # We'll break out of the event_source loop now,
                            # and handle final output below.
                            break
                        else:
                            # Another finish reason that may require no forwarding.
                            should_forward = False

                    # Handle tool calls if present
                    if parsed_data.choices[0].delta.tool_calls is not None:
                        should_forward = False
                        tcall = parsed_data.choices[0].delta.tool_calls[0]
                        assert tcall.function is not None
                        tool_call_name = tcall.function.name or tool_call_name
                        tool_call_id = tcall.id if tcall.id is not None else tool_call_id
                        tool_args = tcall.function.arguments
                        tool_call_json += tool_args if tool_args is not None else ""

                    # Forward SSE messages to the client if allowed
                    if should_forward and sse.data:
                        logger.debug("Forwarding message to client")
                        try:
                            yield SSEData.model_validate_json(sse.data).model_dump_json()
                        except (asyncio.CancelledError, WebSocketDisconnect, ConnectionError):
                            logger.warning("Client disconnected during SSE forwarding.")
                            return

                    # Save the last message
                    last = parsed_data

            # If we got here, we exited the aiter_sse loop either because of [DONE] or a finish reason
            if last is None:
                # No data was ever received, break out (nothing more to do)
                logger.debug("No data received, assuming done.")
                fully_done = True
                break

            # Check finish reason again after exiting the SSE loop
            if last.choices[0].finish_reason is not None:
                if last.choices[0].finish_reason.value in ["stop", "length"]:
                    logger.debug("No more messages to receive. Finishing.")
                    fully_done = True
                else:
                    # Non-standard finish reason
                    logger.debug("Non-standard finish reason, stopping.")
                    fully_done = True

            # If no tool calls found, we're done
            if tool_call_name == "":
                logger.debug("No tool calls found in this cycle.")
                # If we reached here, no more content is coming.
                # fully_done might already be True.
                # If not, set it now.
                fully_done = True

            # If we have a tool call, let's execute it
            if tool_call_name:
                logger.debug(f"Tool calls found: {tool_call_name=} {tool_call_json=}")
                msg = ChatCompletionRequestMessage(
                    role="assistant",
                    content=response_content,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id=tool_call_id,
                            type="function",
                            function=Function1(
                                name=tool_call_name, arguments=tool_call_json
                            ),
                        )
                    ],
                )  # type: ignore
                request.messages.append(msg)

                # Execute the tool call
                session = await ClientManager.get_client_from_tool(tool_call_name)
                tool_call_result = await session.call_tool(
                    name=tool_call_name,
                    arguments=json.loads(tool_call_json),
                )

                logger.debug(
                    f"Tool call result for {tool_call_name}: {tool_call_result.model_dump()}"
                )
                logger.debug(f"Tool call result content: {tool_call_result.content}")
                tools_content = [{"type": "text", "text": part.text} for part in tool_call_result.content]

                request.messages.append(
                    ChatCompletionRequestMessage.model_validate(
                        {
                            "role": "tool",
                            "content": tools_content,
                            "tool_call_id": tool_call_id,
                        }
                    )
                )


                logger.debug("Sending next iteration of chat completion request")
                # This will loop again since fully_done is still False if we expect more output.
                # If no more output is expected, fully_done should be set True before continuing.

    except (asyncio.CancelledError, ConnectionError) as e:
        logger.warning(f"Client disconnected or connection error: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error in chat_completions: {e}")
        return

    # When fully done, send the final DONE event and break
    logger.debug("Streaming complete. Sending final [DONE] event.")
    try:
        yield ServerSentEvent(event="message", data="[DONE]", id=None, retry=None)
    except (asyncio.CancelledError, ConnectionError):
        logger.warning("Client disconnected before receiving final [DONE].")
        return