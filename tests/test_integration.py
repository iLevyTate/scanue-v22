import pytest
from unittest.mock import AsyncMock, patch
from typing import Dict, Any
import asyncio

@pytest.mark.asyncio
async def test_full_processing_chain():
    """Test the entire processing chain with all fixes"""
    test_input = {
        "content": "Calculate 8 * 8\n# Section Header\nRegular text",
        "metadata": {
            "temperature": 0.7,
            "model": "test-model"
        }
    }
    
    async with AsyncAgent() as agent:
        result = await agent.process_full_chain(test_input)
        
        # Verify mathematical expression preservation
        assert "8 * 8" in result["content"]
        
        # Verify section handling
        assert "# Section Header" in result["content"]
        
        # Verify metadata preservation
        assert result["metadata"]["temperature"] == 0.7
        assert result["metadata"]["model"] == "test-model"
        
        # Verify content structure
        assert isinstance(result["content"], str)
        assert "Regular text" in result["content"]

@pytest.mark.asyncio
async def test_concurrent_processing():
    """Test concurrent processing with metadata preservation"""
    test_inputs = [
        {"content": f"Test {i}", "metadata": {"id": i}} 
        for i in range(5)
    ]
    
    async with AsyncAgent() as agent:
        tasks = [
            agent.process(input_data["content"], metadata=input_data["metadata"])
            for input_data in test_inputs
        ]
        
        results = await asyncio.gather(*tasks)
        
        for i, result in enumerate(results):
            assert result["metadata"]["id"] == i 