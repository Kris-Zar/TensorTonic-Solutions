import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # Write code here
    G_new = G + np.square(g)

    w_new = w - (lr/(np.sqrt(G_new+eps))) * g

    return w_new, G_new
    pass