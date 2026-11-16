from unittest.mock import patch

import pytest

from core.app.entities.task_entities import CompletionAppBlockingResponse
from core.model_runtime.entities.llm_entities import LLMResultChunk, LLMResultChunkDelta, LLMUsage
from core.model_runtime.entities.message_entities import AssistantPromptMessage, TextPromptMessageContent
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel


@pytest.mark.parametrize("text", ["", "synthetic answer"])
def test_blocking_completion_preserves_text_even_when_empty(text):
    chunk = LLMResultChunk(model="synthetic", prompt_messages=[], delta=LLMResultChunkDelta(
        index=0, message=AssistantPromptMessage(content=text), usage=LLMUsage.empty_usage()
    ))
    model = LargeLanguageModel.model_construct(tenant_id="synthetic", plugin_id="synthetic", provider_name="synthetic")
    with patch("core.model_runtime.model_providers.__base.large_language_model.PluginModelManager") as manager:
        manager.return_value.invoke_llm.return_value = iter([chunk])
        result = model.invoke(model="synthetic", credentials={}, prompt_messages=[], stream=False)
    response = CompletionAppBlockingResponse.Data(
        id="synthetic", mode="completion", message_id="synthetic", answer=result.message.content, created_at=0
    )
    assert response.answer == text


def test_blocking_multimodal_content_is_preserved():
    part = TextPromptMessageContent(data="synthetic part")
    chunk = LLMResultChunk(model="synthetic", prompt_messages=[], delta=LLMResultChunkDelta(
        index=0, message=AssistantPromptMessage(content=[part]), usage=LLMUsage.empty_usage()
    ))
    model = LargeLanguageModel.model_construct(tenant_id="synthetic", plugin_id="synthetic", provider_name="synthetic")
    with patch("core.model_runtime.model_providers.__base.large_language_model.PluginModelManager") as manager:
        manager.return_value.invoke_llm.return_value = iter([chunk])
        result = model.invoke(model="synthetic", credentials={}, prompt_messages=[], stream=False)
    assert result.message.content == [part]


@pytest.mark.parametrize("content", [None, []])
def test_blocking_tool_only_result_preserves_tool_call(content):
    tool = AssistantPromptMessage.ToolCall(
        id="synthetic-tool", type="function",
        function=AssistantPromptMessage.ToolCall.ToolCallFunction(name="lookup", arguments='{"key":"synthetic"}')
    )
    chunk = LLMResultChunk(model="synthetic", prompt_messages=[], delta=LLMResultChunkDelta(
        index=0, message=AssistantPromptMessage(content=content, tool_calls=[tool]), usage=LLMUsage.empty_usage()
    ))
    model = LargeLanguageModel.model_construct(tenant_id="synthetic", plugin_id="synthetic", provider_name="synthetic")
    with patch("core.model_runtime.model_providers.__base.large_language_model.PluginModelManager") as manager:
        manager.return_value.invoke_llm.return_value = iter([chunk])
        result = model.invoke(model="synthetic", credentials={}, prompt_messages=[], stream=False)
    assert result.message.content == ""
    assert result.message.tool_calls == [tool]
