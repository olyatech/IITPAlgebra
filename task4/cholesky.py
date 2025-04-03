import numpy as np
import sys

def is_symmetric_positive_definite(A, tol=1e-8):
    """
    Check positive definiteness by examining eigenvalues.
    Returns True if all eigenvalues are greater than tolerance.
    """
    if not np.allclose(A, A.T, atol=tol):  # Check symmetry first
        return False
    eigvals = np.linalg.eigvalsh(A)  # For symmetric matrices
    return np.all(eigvals > tol)

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition for symmetric positive definite matrix A = LLᵀ.
    
    Parameters:
        A (numpy.ndarray): Symmetric positive definite matrix
        
    Returns:
        numpy.ndarray: Lower triangular matrix L such that A = LLᵀ
        
    Raises:
        ValueError: If matrix is not symmetric positive definite
    """
    if A.shape[0] != A.shape[1]:
        raise np.linalg.LinAlgError("Matrix A is not square")

    if A.shape[0] != np.linalg.matrix_rank(A):
        raise np.linalg.LinAlgError("Singular matrix A was given")
    
    if not is_symmetric_positive_definite(A):
        raise np.linalg.LinAlgError("Matrix is not symmetric positive definite")
    
    n = A.shape[0]
    L = np.zeros_like(A)
    
    for i in range(n):
        for j in range(i+1):
            if i == j:
                # Diagonal elements
                s = np.sum(L[i, :j]**2)
                L[i, j] = np.sqrt(A[i, i] - s)
            else:
                # Off-diagonal elements
                s = np.sum(L[i, :j] * L[j, :j])
                L[i, j] = (A[i, j] - s) / L[j, j]
    
    return L

def forward_substitution(L, b):
    """
    Solve Ly = b using forward substitution.
    
    Parameters:
        L (numpy.ndarray): Lower triangular matrix
        b (numpy.ndarray): Right-hand side vector
        
    Returns:
        numpy.ndarray: Solution vector y
    """
    n = L.shape[0]
    y = np.zeros(n)
    
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    
    return y

def backward_substitution(L, y):
    """
    Solve Lᵀx = y using backward substitution.
    
    Parameters:
        L (numpy.ndarray): Lower triangular matrix (from Cholesky)
        y (numpy.ndarray): Vector from forward substitution
        
    Returns:
        numpy.ndarray: Solution vector x
    """
    n = L.shape[0]
    x = np.zeros(n)
    
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(L[i+1:, i], x[i+1:])) / L[i, i]
    
    return x

def solve_cholesky(A, b):
    """
    Solve Ax = b using Cholesky decomposition.
    
    Parameters:
        A (numpy.ndarray): Symmetric positive definite matrix
        b (numpy.ndarray): Right-hand side vector
        
    Returns:
        tuple: (solution x, residual norm ||Ax - b||)
        
    Raises:
        ValueError: If matrix is not symmetric positive definite
    """
    if A.shape[0] != A.shape[1]:
        raise np.linalg.LinAlgError("Matrix A is not square")

    if A.shape[0] != b.shape[0]:
        raise np.linalg.LinAlgError("Matrix A and vector b dimension mismatch")

    if A.shape[0] != np.linalg.matrix_rank(A):
        raise np.linalg.LinAlgError("Singular matrix A was given")
    
    if not is_symmetric_positive_definite(A):
        raise np.linalg.LinAlgError("Matrix is not symmetric positive definite")
    
    L = cholesky_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(L, y)
    
    return x

def test_cholesky_solver():
    """
    Test the Cholesky decomposition and linear system solver.
    """
    # Random symmetric positive definite matrix
    matrix_size = 3
    A = np.random.rand(matrix_size, matrix_size)
    A = A.T @ A + np.eye(matrix_size) * 0.1
    print("Random positive definite matrix A:")
    print(A)
    
    # Random right-hand side
    b = np.random.rand(matrix_size)
    print("Random vector b:")
    print(b)
    
    try:
        x = solve_cholesky(A, b)
        residual = np.linalg.norm(A @ x - b)
        print("Solution x:", x)
        print("Residual ||Ax - b||:", residual)
        
        # Verify with numpy's solver
        x_np = np.linalg.solve(A, b)
        print("NumPy solution:", x_np)
        print("Difference:", np.linalg.norm(x - x_np))

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    test_cholesky_solver()