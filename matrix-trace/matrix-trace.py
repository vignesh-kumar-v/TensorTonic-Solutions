import numpy as np

def matrix_trace(A: list) -> float:
    """
    Returns the trace as a float.
    """
    trace = 0.0
    for i in range(len(A)):
        trace += A[i][i]
    return float(trace)