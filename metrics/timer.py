import time
import functools
from typing import Callable, Any, Optional
from contextlib import contextmanager

@contextmanager
def measure_time(metric_name: str, recorder: Optional[Any] = None):
    """
    Context manager for measuring execution time of a code block.
    
    Args:
        metric_name: Name of the metric to record
        recorder: MetricsRecorder instance to use for recording the time
        
    Example:
        with measure_time("processing_time", recorder):
            # Code to measure
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        if recorder is not None:
            recorder.record(metric_name, elapsed)
        else:
            print(f"{metric_name}: {elapsed:.4f}s")

def timed(metric_name: str):
    """
    Decorator to measure execution time of a function.
    
    Args:
        metric_name: Name to use for the timing metric
        
    Example:
        @timed("query_execution")
        def run_query():
            # Function to time
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            recorder = kwargs.get('recorder', None)
            with measure_time(metric_name, recorder):
                return func(*args, **kwargs)
        return wrapper
    return decorator