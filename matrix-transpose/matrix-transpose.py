import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.asarray(A,dtype=float)
    rows,cols = A.shape
    B = np.zeros((cols,rows),dtype=A.dtype)

    for i in range(rows):
        for j in range(cols):
            B[j,i] = A[i,j]
    return B
    pass
