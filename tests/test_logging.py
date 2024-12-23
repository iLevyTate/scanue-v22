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

def test_log_state(test_logger):
    """Test state logging"""
    test_state = {
        "task": "test task",
        "stage": "test_stage",
        "response": "test response"
    }
    
    test_logger.log_state(test_state, "test_stage")
    
    # Check JSON log file exists and contains the entry
    with open(test_logger.json_log_file, 'r') as f:
        log_data = json.load(f)
        assert len(log_data) == 1
        assert log_data[0]["stage"] == "test_stage"
        assert log_data[0]["state"] == test_state

def test_log_feedback(test_logger):
    """Test feedback logging"""
    test_state = {"stage": "test_stage"}
    test_feedback = "test feedback"
    
    test_logger.log_feedback(test_feedback, test_state)
    
    # Check JSON log file exists and contains the entry
    with open(test_logger.json_log_file, 'r') as f:
        log_data = json.load(f)
        assert len(log_data) == 1
        assert log_data[0]["type"] == "feedback"
        assert log_data[0]["feedback"] == test_feedback
        assert log_data[0]["state"] == test_state

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