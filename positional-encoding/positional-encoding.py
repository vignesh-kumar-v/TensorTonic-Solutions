import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    positions = np.expand_dims(np.arange(0, seq_len), axis=1)
    dims = (np.arange(0, np.ceil(d_model/2).astype(int))).reshape((1, -1))
    frequencies = np.exp(((2*dims)/d_model) * np.log(base))
    sin_vals = np.sin(positions / frequencies)
    cos_vals = np.cos(positions / frequencies)
    pos_embeds = (np.stack([sin_vals, cos_vals], axis=-1)).reshape(seq_len, -1)
    return pos_embeds[:, :d_model]