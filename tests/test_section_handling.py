import pytest
from unittest.mock import AsyncMock, patch
from typing import Dict, Any
import asyncio

@pytest.mark.asyncio
async def test_section_header_detection():
    """Test that section headers are correctly identified"""
    test_cases = [
        ("# Header", True),           # Markdown header
        ("Regular text", False),      # Regular text
        ("Text with # inside", False),# Hash in middle
        ("  # Indented", True),      # Indented header
        ("*Bold text*", False)        # Other markdown
    ]
    
    for input_text, should_be_header in test_cases:
        is_header = is_section_header(input_text)
        assert is_header == should_be_header, f"Failed header detection for: {input_text}"

@pytest.mark.asyncio
async def test_content_type_handling():
    """Test handling of different content types"""
    test_cases = [
        ("Simple string", str),
        (["List", "of", "strings"], list),
        ({"key": "value"}, dict),
        ([{"key": "value"}, "string"], list),
    ]
    
    for content, expected_type in test_cases:
        result = await process_content(content)
        assert isinstance(result, expected_type) 