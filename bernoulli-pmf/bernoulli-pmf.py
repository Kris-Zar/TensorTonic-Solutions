import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    # Write code here
    x = np.asarray(x,dtype= float)

    pmf = [p if i==1 else (1-p) for i in x]
    pmf = np.asarray(pmf,dtype= float)

    mean = p
    var = (p*(1-p))
    return (pmf,mean,var)
    pass