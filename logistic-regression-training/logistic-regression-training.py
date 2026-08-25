import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def Binary_Cross_Entropy(predictions, targets):
    loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    return loss

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    num_of_samples, num_of_features = X.shape
    w = np.zeros(num_of_features)
    b = 0.0
    for epoch in range(steps):
        predictions = _sigmoid(X @ w + b)
        loss = Binary_Cross_Entropy(predictions, y)
        w_grad = (X.T @ (predictions - y)) / num_of_samples
        b_grad = np.mean(predictions - y)
        w -= lr * w_grad
        b -= lr * b_grad
    return (w, b)