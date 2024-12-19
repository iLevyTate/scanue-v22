import pytest
import os
import json
from utils.logging import InteractionLogger
from typing import Dict, Any

@pytest.fixture
def test_logger(tmp_path):
    """Create a test logger that writes to a temporary directory"""
    return InteractionLogger(log_dir=str(tmp_path))

def test_logger_initialization(tmp_path):
    """Test logger initialization and directory creation"""
    logger = InteractionLogger(log_dir=str(tmp_path))
    assert os.path.exists(logger.log_dir)
    assert logger.session_id is not None
    assert os.path.exists(logger.json_log_file)

def test_log_state_with_stage_transition(test_logger):
    """Test logging state transitions"""
    initial_state = {
        "task": "test task",
        "stage": "task_delegation",
        "response": ""
    }
    updated_state = {
        "task": "test task",
        "stage": "emotional_regulation",
        "response": "test response"
    }
    
    test_logger.log_state(initial_state, "task_delegation")
    test_logger.log_state(updated_state, "emotional_regulation")
    
    with open(test_logger.json_log_file, 'r') as f:
        log_data = json.load(f)
        assert len(log_data) == 2
        assert log_data[0]["stage"] == "task_delegation"
        assert log_data[1]["stage"] == "emotional_regulation"
        assert "timestamp" in log_data[0]
        assert "timestamp" in log_data[1]

def test_log_error_state(test_logger):
    """Test logging error states"""
    error_state = {
        "error": True,
        "response": "Test error occurred",
        "stage": "error"
    }
    
    test_logger.log_state(error_state, "error")
    
    with open(test_logger.json_log_file, 'r') as f:
        log_data = json.load(f)
        assert len(log_data) == 1
        assert log_data[0]["state"]["error"] is True 

def test_log_state_with_scanaq(test_logger):
    """Test logging state with SCANAQ results"""
    test_state = {
        "task": "test task",
        "stage": "task_delegation",
        "scanaq_results": "Test SCANAQ results"
    }
    
    test_logger.log_state(test_state, "task_delegation")
    
    with open(test_logger.json_log_file, 'r') as f:
        log_data = json.load(f)
        assert len(log_data) == 1
        assert "scanaq_results" in log_data[0]["state"]

def test_log_feedback_with_history(test_logger):
    """Test logging feedback with history"""
    state = {
        "task": "test task",
        "stage": "value_assessment",
        "feedback_history": [
            {"feedback": "previous feedback"}
        ]
    }
    
    test_logger.log_feedback("new feedback", state)
    
    with open(test_logger.json_log_file, 'r') as f:
        log_data = json.load(f)
        assert len(log_data) == 1
        assert log_data[0]["feedback"] == "new feedback"
        assert "feedback_history" in log_data[0]["state"]