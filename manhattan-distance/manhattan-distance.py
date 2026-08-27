import numpy as np

def manhattan_distance(x: list, y: list) -> float:
    """
    Returns the Manhattan distance as a Python float.
    """
    x = np.array(x)
    y = np.array(y)
    return np.sum(np.abs(x - y)).astype(float)