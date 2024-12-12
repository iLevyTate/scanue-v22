import pytest
from unittest.mock import AsyncMock, patch
from typing import Dict, Any
import asyncio

@pytest.mark.asyncio
async def test_error_with_metadata():
    """Test that metadata is preserved during errors"""
    test_metadata = {"important": "data"}
    
    with pytest.raises(ProcessingError) as exc_info:
        async with AsyncAgent() as agent:
            await agent.process(
                content="error triggering content",
                metadata=test_metadata,
                should_fail=True
            )
    
    assert exc_info.value.metadata == test_metadata

@pytest.mark.asyncio
async def test_invoke_method():
    """Test the new invoke method implementation"""
    test_cases = [
        ("Simple input", None),
        ("Input with metadata", {"temperature": 0.7}),
        ("Complex input", {"nested": {"key": "value"}}),
    ]
    
    for content, metadata in test_cases:
        result = await invoke_with_metadata(content, metadata)
        if metadata:
            assert result["metadata"] == metadata 

@pytest.mark.asyncio
async def test_error_propagation():
    """Test error propagation through agent chain"""
    test_cases = [
        (TimeoutError("Agent timeout"), "timeout"),
        (ConnectionError("Network error"), "connection"),
        (ValueError("Invalid state"), "processing"),
        (AuthenticationError("Invalid key"), "connection"),
        (APIError("Rate limit"), "connection")
    ]
    
    for error, expected_type in test_cases:
        with patch("agents.base.BaseAgent._process_with_timeout", side_effect=error):
            result = await process_full_chain({"task": "test"})
            assert result["error"]
            assert result["error_type"] == expected_type
            assert "metadata" in result
            assert "original_error" in result["metadata"]