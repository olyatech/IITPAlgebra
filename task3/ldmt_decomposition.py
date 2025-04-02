import numpy as np

def ldmt_decomposition(A):
    """
    Perform LDM^T decomposition for a symmetric positive definite matrix A.
    
    Parameters:
        A (numpy.ndarray): Square symmetric positive definite matrix
        
    Returns:
        L, D, M (numpy.ndarray): Matrices of the decomposition A = L @ D @ M.T
    """
    if A.shape[0] != A.shape[1]:
        raise np.linalg.LinAlgError("Matrix A is not square")
    
    if A.shape[0] != np.linalg.matrix_rank(A):
        raise np.linalg.LinAlgError("Singular matrix A was given")
    
    n = A.shape[0]
    L = np.eye(n) 
    D = np.zeros((n, n))  
    M = np.eye(n)  
    
    for j in range(n):
        # Compute D[j,j]
        sum_d = 0.0
        for k in range(j):
            sum_d += L[j,k] * D[k,k] * M[j,k]
        D[j,j] = A[j,j] - sum_d
        
        # Compute L[i,j] and M[i,j] for i > j
        for i in range(j+1, n):
            sum_l = 0.0
            for k in range(j):
                sum_l += L[i,k] * D[k,k] * M[j,k]
            L[i,j] = (A[i,j] - sum_l) / D[j,j]
            
            sum_m = 0.0
            for k in range(j):
                sum_m += L[j,k] * D[k,k] * M[i,k]
            M[i,j] = (A[j,i] - sum_m) / D[j,j]
    
    return L, D, M

def is_positive_definite(A):
    """
    Check if a matrix is positive definite using LDM^T decomposition.

    Parameters:
    A (numpy.ndarray): The input matrix to check. Must be a square matrix.

    Returns:
    bool: True if the matrix is positive definite, False otherwise.
    """
    if A.shape[0] != A.shape[1]:
        raise np.linalg.LinAlgError("Input matrix A must be square.")
    
    if A.shape[0] != np.linalg.matrix_rank(A):
        raise np.linalg.LinAlgError("Singular matrix A was given")
    
    L, D, M = ldmt_decomposition(A)
    return np.all(np.diag(D) > 0)

def example_usage():
    matrix_size = 3
    A = np.random.rand(matrix_size, matrix_size)
    print("A:\n", A)

    L, D, M = ldmt_decomposition(A)
    print("L:\n", L)
    print("D:\n", D)
    print("M:\n", M)
    print("LDM^T:\n", L @ D @ M.T)

    A_is_positive_definite = is_positive_definite(A)
    print("A is positive definite:", A_is_positive_definite)

if __name__ == "__main__":
    example_usage()
