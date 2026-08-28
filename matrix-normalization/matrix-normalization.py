import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    values = np.array(matrix, dtype=float)
    if norm_type == "l1":
        norms = np.sum(np.abs(values), axis=axis, keepdims=True)
    elif norm_type == "l2":
        norms = np.sqrt(np.sum(np.square(values), axis=axis, keepdims=True))
    elif norm_type == "max":
        norms = np.max(np.abs(values), axis=axis, keepdims=True)
    values = np.divide(values, norms, out=np.zeros_like(values), where=norms != 0,)
    return values