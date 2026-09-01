import numpy as np

def manhattan_distance(x: list, y: list) -> float:
    """
    Returns the Manhattan distance as a Python float.
    """
    # Write code here
    x = np.asarray(x,dtype=float)
    y = np.asarray(y,dtype=float)
    return np.abs(x-y).sum()
    pass