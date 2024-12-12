import pytest
from unittest.mock import AsyncMock, patch
from typing import Dict, Any
import asyncio

@pytest.mark.asyncio
async def test_mathematical_expression_preservation():
    """Test that mathematical expressions are preserved"""
    test_cases = [
        ("Calculate 8 * 8", "Calculate 8 * 8"),
        ("2 * (3 + 4) * 5", "2 * (3 + 4) * 5"),  # Added nested operations
        ("x*y + z/(a-b)", "x*y + z/(a-b)"),      # Added complex expressions
        ("8*8=64", "8*8=64"),
        ("Area = π*r²", "Area = π*r²"),
        ("∑(x_i * y_i)", "∑(x_i * y_i)"),        # Added mathematical symbols
        ("lim_{x→∞}", "lim_{x→∞}")              # Added edge cases
    ]
    
    for input_text, expected in test_cases:
        result = await process_content(input_text)
        assert result == expected, f"Failed to preserve: {input_text}"

@pytest.mark.asyncio
async def test_kwargs_preservation():
    """Test that kwargs/metadata is preserved through processing"""
    test_metadata = {
        "temperature": 0.7,
        "model": "test-model",
        "nested": {"key": "value"}
    }
    
    async with AsyncAgent() as agent:
        result = await agent.process(
            content="test content",
            metadata=test_metadata
        )
        
        assert "metadata" in result
        assert result["metadata"] == test_metadata
        assert result["metadata"]["temperature"] == 0.7
        assert result["metadata"]["nested"]["key"] == "value" 