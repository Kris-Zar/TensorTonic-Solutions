import math 
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    eula = []
    for i in x:
        if i > 0:
            eula.append(i)
        else:
            eula.append(alpha*(math.exp(i)-1))
    return eula