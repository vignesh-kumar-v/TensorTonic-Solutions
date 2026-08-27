import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a Python float.
    """
    x, y = np.array(x), np.array(y)
    return np.sqrt((np.sum(np.square(x - y)))).astype(float)