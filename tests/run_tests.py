import pytest
import asyncio
import os
from dotenv import load_dotenv
from test_suite import TestSuite

def setup_environment():
    """Setup test environment"""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "test-key"

async def main():
    """Run all tests and report results"""
    print("Starting comprehensive test suite...")
    setup_environment()
    
    try:
        # Create fixtures
        env_fixture = TestSuite.mock_env_vars()
        llm_fixture = TestSuite.mock_llm()
        
        # Run tests with fixtures
        async with env_fixture, llm_fixture:
            await TestSuite.test_response_handling(env_fixture, llm_fixture)
            await TestSuite.test_concurrent_processing(env_fixture, llm_fixture)
        print("All tests passed successfully!")
    except Exception as e:
        print(f"Test suite failed: {str(e)}")
        raise

if __name__ == "__main__":
    pytest.main(["-v", "test_suite.py"]) 