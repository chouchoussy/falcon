"""
Log Parser module for FALCON.
Parses execution trace log files to extract function call events.

Key Features:
- Extracts function calls with filename, function name, and thread ID
- Supports multi-threaded logs: [Thread-123] Call func_name ...
- Handles single-threaded logs (defaults to "main" thread)

Thread Handling (Critical for FALCON):
As per FALCON paper, sequential edges should ONLY connect methods
within the same thread. This parser extracts thread IDs to ensure
correct intra-thread execution sequence reconstruction.
"""

import re
from typing import List, Dict, Tuple
from pathlib import Path


class LogEvent:
    """Represents a single log event (function call)."""
    
    def __init__(self, filename: str, funcname: str, thread_id: str = "main"):
        self.filename = filename
        self.funcname = funcname
        self.thread_id = thread_id
        self.unique_id = f"{filename}::{funcname}"
    
    def __repr__(self):
        return f"LogEvent({self.unique_id}, thread={self.thread_id})"
    
    def to_dict(self) -> Dict:
        return {
            "filename": self.filename,
            "funcname": self.funcname,
            "thread_id": self.thread_id,
            "unique_id": self.unique_id
        }


def parse_log_file(filepath: str) -> List[LogEvent]:
    """
    Parse a log file and extract function call events.
    
    Args:
        filepath: Path to the log file
    
    Returns:
        List of LogEvent objects
    
    Example log line:
        Call function_name from source (filename.c:123)
    """
    # Regex pattern to extract function name and filename
    # Pattern: Call <func_name> ... (filename.ext:line_number)
    # Pattern supports optional thread ID: [Thread-123] Call func_name ...
    # Group 1: Thread ID (optional)
    # Group 2: Function name
    # Group 3: Filename
    pattern = r'(?:\[([^\]]+)\])?\s*Call\s+([^\s]+)\s+.*?\(([\w\-\.]+):\d+\)'
    
    events = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                match = re.search(pattern, line)
                if match:
                    # Extract thread ID (if present in log), function name, and filename
                    thread_id = match.group(1) if match.group(1) else "main"
                    func_name = match.group(2)
                    filename = match.group(3)
                    
                    event = LogEvent(
                        filename=filename,
                        funcname=func_name,
                        thread_id=thread_id
                    )
                    events.append(event)
    
    except FileNotFoundError:
        print(f"Warning: Log file not found: {filepath}")
        return []
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []
    
    return events


def parse_log_directory(directory: str) -> Dict[str, List[LogEvent]]:
    """
    Parse all log files in a directory.
    
    Args:
        directory: Path to directory containing log files
    
    Returns:
        Dictionary mapping test_id -> List[LogEvent]
    """
    log_dir = Path(directory)
    
    if not log_dir.exists():
        print(f"Warning: Directory not found: {directory}")
        return {}
    
    results = {}
    
    # Find all .log files
    log_files = list(log_dir.glob("**/*.log"))
    
    for log_file in log_files:
        # Use stem as test_id (filename without extension)
        test_id = log_file.stem
        events = parse_log_file(str(log_file))
        
        if events:
            results[test_id] = events
    
    return results


def get_unique_functions(events: List[LogEvent]) -> List[str]:
    """
    Extract unique function identifiers from events.
    
    Args:
        events: List of LogEvent objects
    
    Returns:
        List of unique function IDs (filename::funcname)
    """
    unique_ids = set()
    
    for event in events:
        unique_ids.add(event.unique_id)
    
    return sorted(list(unique_ids))


def get_call_sequence(events: List[LogEvent], thread_id: str = None) -> List[str]:
    """
    Get the sequence of function calls, optionally filtered by thread.
    
    Args:
        events: List of LogEvent objects
        thread_id: Optional thread ID to filter by
    
    Returns:
        List of function unique IDs in call order
    """
    if thread_id:
        filtered_events = [e for e in events if e.thread_id == thread_id]
    else:
        filtered_events = events
    
    return [e.unique_id for e in filtered_events]


if __name__ == "__main__":
    # Test the parser
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        events = parse_log_file(test_file)
        print(f"Parsed {len(events)} events from {test_file}")
        
        if events:
            print("\nFirst 5 events:")
            for event in events[:5]:
                print(f"  {event}")
            
            unique_funcs = get_unique_functions(events)
            print(f"\nTotal unique functions: {len(unique_funcs)}")
    else:
        print("Usage: python log_parser.py <log_file>")

