import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    a, b = np.array(a), np.array(b)
    a_norm = np.sqrt(np.sum(np.square(a)))
    b_norm = np.sqrt(np.sum(np.square(b)))
    if a_norm==0 or b_norm==0:
        return 0.0
    else:
        return float(np.dot(a, b)/(a_norm * b_norm))