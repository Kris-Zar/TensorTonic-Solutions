import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Return the Shannon entropy of the class labels.
    """
    # Write code here
    y = np.asarray(y,dtype= float)

    a,counts= np.unique(y, return_counts= True)
    prob = np.asarray([c/len(y) for c in counts])
    H = 0
    
    for i in range(len(counts)):
        H= H + (prob[i]*(np.log2(prob[i])))
    
    return float(-H)
    pass