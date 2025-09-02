"""Utility functions for the multi-agent financial KG construction system."""

import yaml
import os
import time
import threading
import sys
from .chunking import chunk_large_text

def load_prompts(prompt: str = "default"):
    """Load prompts from YAML file using config-based paths."""
    from ma_finkg.config import Config
    prompts_path = Config.get_prompts_file(prompt)
    with open(prompts_path, 'r') as f:
        return yaml.safe_load(f)

def spinner(message="Processing"):
    """Simple spinner context manager."""
    return Spinner(message)

def print_progress(message, final=False):
    """Print message in-place or on new line."""
    if final:
        print(f"\r{message.ljust(100)}")  # Clear line with more padding
    else:
        print(f"\r{message.ljust(100)}", end="", flush=True)  # Clear line with more padding

class Spinner:
    def __init__(self, message, global_start_time=None):
        self.message = message
        self.running = False
        self.thread = None
        self.start_time = None
        self.global_start_time = global_start_time or time.time()
    
    def __enter__(self):
        self.start_time = time.time()
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()
        return self
    
    def __exit__(self, *args):
        self.running = False
        self.thread.join()
        global_elapsed = time.time() - self.global_start_time
        print(f"\r[{global_elapsed:.1f}s] {self.message}... done".ljust(100), end="", flush=True)
    
    def _spin(self):
        chars = "|/-\\"
        i = 0
        while self.running:
            global_elapsed = time.time() - self.global_start_time
            print(f"\r[{global_elapsed:.1f}s] {self.message}... {chars[i % 4]}".ljust(100), end="", flush=True)
            time.sleep(0.1)
            i += 1

# Global timer for consistent timing across all operations
_global_start_time = None

def set_global_timer():
    """Set the global start time for consistent timing."""
    global _global_start_time
    _global_start_time = time.time()
    return _global_start_time

def get_elapsed_time():
    """Get elapsed time since global timer started."""
    if _global_start_time:
        return time.time() - _global_start_time
    return 0.0

def spinner(message="Processing"):
    """Simple spinner context manager with global timing."""
    return Spinner(message, _global_start_time)


