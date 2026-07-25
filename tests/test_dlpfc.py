import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from agents.dlpfc import DLPFCAgent
from typing import Dict, Any

@pytest.fixture
def mock_env_vars():
    # Mock ConfigLoader to avoid reading real config file
    with patch.dict("os.environ", {
        "DLPFC_MODEL": "dlpfc-model",
        "OPENAI_API_KEY": "test-key"
    }), patch("utils.config.ConfigLoader.get_agent_config", return_value={
        "models": {
            "primary": {"provider": "openai", "name": "dlpfc-model"},
            "fast": {"provider": "openai", "name": "fast-model"}
        }
    }):
        yield

@pytest.fixture
def test_state():
    return {
        "task": "test task",
        "stage": "test_stage",
        "response": "",
        "subtasks": [],
        "feedback": "test feedback",
        "previous_response": "previous test response",
        "feedback_history": [
            {
                "stage": "stage1",
                "response": "response1",
                "feedback": "feedback1"
            },
            {
                "stage": "stage2",
                "response": "response2",
                "feedback": "feedback2"
            }
        ],
        "error": False
    }

@pytest.fixture
def dlpfc_agent(mock_env_vars):
    return DLPFCAgent()

@pytest.mark.asyncio
async def test_dlpfc_agent_initialization(dlpfc_agent):
    """Test DLPFC agent initialization"""
    assert isinstance(dlpfc_agent, DLPFCAgent)
    assert dlpfc_agent.llm.model_name == "dlpfc-model"

@pytest.mark.asyncio
async def test_dlpfc_agent_process(dlpfc_agent, test_state):
    """Test DLPFC agent processing"""
    mock_response = AsyncMock()
    mock_response.content = """
    Here's the task breakdown:
    1. Subtask 1 - Assign to VMPFC
    2. Subtask 2 - Assign to OFC
    3. Subtask 3 - Assign to ACC
    
    Integration plan:
    1. Collect responses
    2. Analyze results
    3. Generate final output
    """
    
    dlpfc_agent.llm = AsyncMock()
    dlpfc_agent.llm.ainvoke = AsyncMock(return_value=mock_response)
    
    result = await dlpfc_agent.process(test_state)
    assert isinstance(result, dict)
    assert "subtasks" in result
    assert result["stage"] == "task_delegation"
    assert not result.get("error", False)

@pytest.mark.asyncio
async def test_dlpfc_agent_error_handling(dlpfc_agent, test_state):
    """Test error handling in DLPFC agent"""
    dlpfc_agent.llm = AsyncMock()
    dlpfc_agent.llm.ainvoke = AsyncMock(side_effect=ValueError("Test error"))
    
    result = await dlpfc_agent.process(test_state)
    assert result["error"]
    
    # Check response content
    response = result["response"]
    if isinstance(response, dict):
        response_text = response["content"]
    else:
        response_text = response
        
    assert "error" in response_text.lower()

@pytest.mark.asyncio
async def test_dlpfc_agent_timeout(dlpfc_agent, test_state):
    """Test timeout handling in DLPFC agent"""
    async def mock_process(*args, **kwargs):
        await asyncio.sleep(1)
        return None

    dlpfc_agent.process = mock_process
    
    with pytest.raises(asyncio.TimeoutError):
        async with asyncio.timeout(0.001):
            await dlpfc_agent.process(test_state)

SPEC_FORMAT_REPLY = """**AGENT DELEGATION:**
- VMPFC Agent: YES - emotionally loaded
- OFC Agent: NO
- ACC Agent: YES - the stated goals conflict
- MPFC Agent: YES - Always needed for final integration

**Analysis:**
The financial gap is modest but the wellbeing gain compounds.

**Subtask Breakdown:**
1. Weigh the personal impact of the pay cut
2. Surface the contradiction between stated goals
"""


def test_delegation_block_is_not_parsed_as_subtasks(dlpfc_agent):
    """The AGENT DELEGATION block holds routing decisions, not work items.

    Without skipping it, "- VMPFC Agent: YES - emotionally loaded" became a
    subtask literally named "YES - emotionally loaded", and a bare "NO" became
    another. This only became user-visible once subtasks started being
    propagated into workflow state.
    """
    subtasks = dlpfc_agent._parse_subtasks(SPEC_FORMAT_REPLY)
    texts = [s["task"] for s in subtasks]

    assert texts == [
        "Weigh the personal impact of the pay cut",
        "Surface the contradiction between stated goals",
    ]
    assert not any(t.strip() in {"NO", "YES"} or t.startswith(("YES", "NO")) for t in texts)


def test_unknown_markdown_header_is_not_emitted_as_a_bullet(dlpfc_agent):
    """"**Analysis:**" starts with '*', so it fell through to the bullet branch
    and was rendered as a content bullet under the previous section's heading
    (e.g. "🔄 Integration Plan: • Analysis:**")."""
    content = dlpfc_agent._format_response(SPEC_FORMAT_REPLY)["response"]["content"]

    assert "Analysis:**" not in content
    assert "• Analysis" not in content


def test_prose_only_reply_is_not_summarized_into_nothing(dlpfc_agent):
    """A reply with none of the expected sections used to yield empty content."""
    prose = "I recommend taking the job; the hours matter more than the pay gap."
    content = dlpfc_agent._format_response(prose)["response"]["content"]

    assert content.strip() == prose


@pytest.mark.asyncio
async def test_dlpfc_applies_inner_llm_timeout(dlpfc_agent, test_state):
    """C8: DLPFC overrides process() and used to call self.llm.ainvoke directly,
    bypassing AGENT_LLM_TIMEOUT_SECONDS. That made the 'inner timeout fires
    before the outer node timeout' invariant vacuous for the one agent that runs
    on every single task."""
    async def slow_ainvoke(*args, **kwargs):
        await asyncio.sleep(5)

    dlpfc_agent.llm = AsyncMock()
    dlpfc_agent.llm.ainvoke = slow_ainvoke

    dlpfc_agent.llm_timeout = 0.01
    result = await dlpfc_agent.process(test_state)

    assert result["error"] is True
    assert "timed out" in result["response"]["content"].lower()


@pytest.mark.asyncio
async def test_dlpfc_agent_cancellation_propagates(dlpfc_agent, test_state):
    """CancelledError must propagate out of DLPFC.process (no longer swallowed)."""
    dlpfc_agent.llm = AsyncMock()
    dlpfc_agent.llm.ainvoke = AsyncMock(side_effect=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await dlpfc_agent.process(test_state)

def test_dlpfc_format_feedback_history(dlpfc_agent, test_state):
    """Test feedback history formatting"""
    history = test_state["feedback_history"]
    formatted = dlpfc_agent._format_feedback_history(history)
    assert "stage1" in formatted
    assert "response1" in formatted
    assert "feedback1" in formatted
    assert "stage2" in formatted
    assert "response2" in formatted
    assert "feedback2" in formatted

def test_dlpfc_format_feedback_history_empty(dlpfc_agent):
    """Test feedback history formatting with empty history"""
    formatted = dlpfc_agent._format_feedback_history([])
    assert formatted == "No previous feedback"

@pytest.mark.asyncio
async def test_dlpfc_parse_subtasks(dlpfc_agent):
    """Test subtask parsing"""
    response = """
    Here's the task breakdown:
    1. Analyze data - Assign to VMPFC
    2. Generate options - Assign to OFC
    3. Monitor progress - Assign to ACC
    """
    subtasks = dlpfc_agent._parse_subtasks(response)
    assert isinstance(subtasks, list)
    assert len(subtasks) > 0
    assert all(isinstance(task, dict) for task in subtasks)
    assert all("task" in task and "agent" in task for task in subtasks)

@pytest.mark.asyncio
async def test_malformed_llm_response(dlpfc_agent, test_state):
    """Test handling of malformed LLM responses."""
    malformed_responses = [
        # Empty response
        "",
        # Invalid JSON
        "{invalid json}",
        # Missing required sections
        "Random text without structure",
        # Incomplete structure
        """
        Here's the task breakdown:
        1. Incomplete task
        Integration plan:
        """,
        # Invalid task format
        """
        Here's the task breakdown:
        Invalid task format
        No proper numbering or structure
        """
    ]
    
    for response in malformed_responses:
        mock_response = AsyncMock()
        mock_response.content = response
        dlpfc_agent.llm = AsyncMock()
        dlpfc_agent.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        result = await dlpfc_agent.process(test_state)
        assert isinstance(result, dict)
        assert "subtasks" in result
        assert isinstance(result["subtasks"], list)
        # Should handle malformed input gracefully
        assert not result.get("error", False)

@pytest.mark.asyncio
async def test_complex_subtask_assignments(dlpfc_agent):
    """Test parsing of complex subtask assignments with nested structure."""
    complex_response = """
    Task Breakdown:
    1. Main Task A
        1.1 Subtask A1 - Assign to VMPFC
        1.2 Subtask A2 - Assign to OFC
    2. Main Task B
        2.1 Subtask B1
            - First part - Assign to ACC
            - Second part - Assign to MPFC
        2.2 Subtask B2 - Assign to VMPFC
    
    Integration Steps:
    1. Collect all subtask results
    2. Analyze dependencies
    3. Generate final report
    """
    
    subtasks = dlpfc_agent._parse_subtasks(complex_response)
    assert isinstance(subtasks, list)
    assert len(subtasks) > 0
    
    # Verify structure handling
    tasks = [task["task"] for task in subtasks]
    assert any("Main Task A" in task for task in tasks)
    assert any("Subtask A1" in task for task in tasks)
    assert any("Subtask B1" in task for task in tasks)
    
    # Verify agent assignments
    agent_assignments = [task["agent"] for task in subtasks if task["agent"]]
    assert "VMPFC" in agent_assignments
    assert "OFC" in agent_assignments
    assert "ACC" in agent_assignments
    assert "MPFC" in agent_assignments

@pytest.mark.asyncio
async def test_invalid_feedback_history(dlpfc_agent):
    """Test handling of invalid feedback history formats."""
    invalid_histories = [
        # Empty list
        [],
        # Missing required fields
        [{"stage": "stage1"}],
        # Invalid types
        [{"stage": 123, "response": 456, "feedback": 789}],
        # None values
        [{"stage": None, "response": None, "feedback": None}],
        # Extra fields
        [{"stage": "stage1", "response": "resp1", "feedback": "feed1", "extra": "field"}]
    ]
    
    for history in invalid_histories:
        formatted = dlpfc_agent._format_feedback_history(history)
        assert isinstance(formatted, str)
        if not history:
            assert formatted == "No previous feedback"
        else:
            assert "stage" in formatted.lower()
            assert "response" in formatted.lower()
            assert "feedback" in formatted.lower()

@pytest.mark.asyncio
async def test_concurrent_subtask_processing(dlpfc_agent):
    """Test handling of concurrent subtask processing."""
    mock_response = AsyncMock()
    mock_response.content = """
    Here's the task breakdown:
    1. Parallel Task 1 - Assign to VMPFC
    2. Parallel Task 2 - Assign to OFC
    3. Parallel Task 3 - Assign to ACC
    
    All tasks can be processed concurrently.
    """
    
    dlpfc_agent.llm = AsyncMock()
    dlpfc_agent.llm.ainvoke = AsyncMock(return_value=mock_response)
    
    result = await dlpfc_agent.process({"task": "concurrent test"})
    assert isinstance(result, dict)
    assert "subtasks" in result
    assert len(result["subtasks"]) == 3
    
    # Verify each task has proper assignment
    agents = [task["agent"] for task in result["subtasks"] if task["agent"]]
    assert len(agents) == 3
    assert set(agents) == {"VMPFC", "OFC", "ACC"}

@pytest.mark.asyncio
async def test_response_formatting_edge_cases(dlpfc_agent):
    """Test edge cases in response formatting."""
    edge_case_responses = [
        # Mixed formatting
        """
        **Task Breakdown:**
        1. *Task 1* - Assign to VMPFC
        2. __Task 2__ - Assign to OFC
        
        # Integration Plan
        * Step 1
        * Step 2
        """,
        # Unicode characters
        """
        📋 Tasks:
        1️⃣ Task 1 - Assign to VMPFC
        2️⃣ Task 2 - Assign to OFC
        
        🔄 Integration:
        ⭐ Step 1
        ⭐ Step 2
        """,
        # HTML-like formatting
        """
        <h1>Task Breakdown:</h1>
        <ul>
        <li>Task 1 - Assign to VMPFC</li>
        <li>Task 2 - Assign to OFC</li>
        </ul>
        """
    ]
    
    for response in edge_case_responses:
        formatted = dlpfc_agent._format_response(response)
        assert isinstance(formatted, dict)
        assert "response" in formatted
        assert not formatted["error"]
        # Response should be a dict with content
        assert isinstance(formatted["response"], dict)
        assert "content" in formatted["response"]
        assert isinstance(formatted["response"]["content"], str)
