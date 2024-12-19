import logging
import json
from datetime import datetime
import os
from typing import Dict, Any
import fcntl

class InteractionLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.interaction_log = []
        
        try:
            self.setup_logging()
            self.initialize_json_log()
            self._load_existing_log()
        except Exception as e:
            logging.error(f"Failed to initialize logger: {str(e)}")
            raise

    def _load_existing_log(self):
        """Load existing log file if it exists for recovery"""
        try:
            if os.path.exists(self.json_log_file):
                with open(self.json_log_file, 'r') as f:
                    self.interaction_log = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load existing log: {str(e)}")
            self.interaction_log = []

    def _safe_write(self, data: list):
        """Thread-safe write with file locking"""
        backup_file = f"{self.json_log_file}.bak"
        try:
            with open(self.json_log_file, 'r+') as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    # Create backup
                    with open(backup_file, 'w') as backup:
                        json.dump(data, backup, indent=2)
                    
                    # Write new data
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    
            # Remove backup if successful
            if os.path.exists(backup_file):
                os.remove(backup_file)
        except Exception as e:
            # Restore from backup if write failed
            if os.path.exists(backup_file):
                os.replace(backup_file, self.json_log_file)
            raise e

    def initialize_json_log(self):
        """Initialize JSON log file with proper error handling"""
        self.json_log_file = os.path.join(
            self.log_dir, 
            f"interaction_{self.session_id}.json"
        )
        try:
            with open(self.json_log_file, 'w') as f:
                json.dump([], f)
        except Exception as e:
            logging.error(f"Failed to initialize JSON log: {str(e)}")
            raise

    def setup_logging(self):
        """Setup logging configuration"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # Setup file handler
        log_file = os.path.join(self.log_dir, f"interaction_{self.session_id}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        # Setup JSON file for structured logging
        self.json_log_file = os.path.join(self.log_dir, f"interaction_{self.session_id}.json")
        
        # Configure logger
        self.logger = logging.getLogger(f"interaction_{self.session_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        
        # Initialize JSON log
        self.interaction_log = []
    
    def log_state(self, state: Dict[str, Any], stage: str):
        """Log state with improved error handling and recovery"""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                "timestamp": timestamp,
                "stage": stage,
                "state": state,
                "type": "state"
            }
            
            self.logger.info(f"Stage: {stage}")
            self.logger.info(f"State: {json.dumps(state, indent=2)}")
            
            self.interaction_log.append(log_entry)
            self._safe_write(self.interaction_log)
        except Exception as e:
            self.logger.error(f"Failed to log state: {str(e)}")
            # Continue execution but log the error
    
    def log_feedback(self, feedback: str, state: Dict[str, Any]):
        """Log user feedback"""
        timestamp = datetime.now().isoformat()
        
        # Create log entry
        log_entry = {
            "timestamp": timestamp,
            "type": "feedback",
            "feedback": feedback,
            "state": state
        }
        
        # Log to file
        self.logger.info(f"Feedback received: {feedback}")
        
        # Add to JSON log
        self.interaction_log.append(log_entry)
        
        # Save JSON log
        with open(self.json_log_file, 'w') as f:
            json.dump(self.interaction_log, f, indent=2) 