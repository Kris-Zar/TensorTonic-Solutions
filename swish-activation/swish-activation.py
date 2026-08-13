import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x = np.asarray(x,dtype=float)
    alpha = 1/(1+np.exp(-x))
    swish = x*alpha
    return swish
    
    pass