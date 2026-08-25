import numpy as np

def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    X, W, b = np.array(X), np.array(W), np.array(b)
    Y = X @ W + b
    Y = Y.tolist()
    return Y