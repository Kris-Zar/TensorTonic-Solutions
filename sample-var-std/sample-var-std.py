import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    x_arr = np.asarray(x, dtype=float)
    x_cap = np.mean(x_arr)

    s_2 = float(
        np.sum(np.square(x_arr - x_cap)) / (len(x_arr) - 1)
    )

    s = float(np.sqrt(s_2))

    return {
        "variance": s_2,
        "standard_deviation": s
    }