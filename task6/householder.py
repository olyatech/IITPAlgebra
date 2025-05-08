import numpy as np
from typing import Tuple

def householder_reflection(a: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Computes the Householder reflection to transform vector a to basis vector e1.
    
    Parameters:
        a (np.ndarray): Input column vector
    
    Returns:
        Tuple[np.ndarray, float]: Reflection vector v and coefficient beta for constructing the reflection matrix
    """
    norm_a = np.linalg.norm(a)
    v = a.copy()
    v[0] += np.sign(a[0]) * norm_a  # Add sign to reduce rounding errors
    beta = 2 / (v.T @ v)
    return v, beta

def qr_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs QR decomposition of matrix A using Householder reflections.
    
    Parameters:
        A (np.ndarray): Full-rank rectangular matrix (m x n, m >= n)
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Matrices Q and R
    """
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    
    for k in range(n):
        # Select subvector for transformation
        a = R[k:, k]
        if np.allclose(a[1:], 0):
            continue  # Already in desired form
        
        # Construct Householder reflection
        v, beta = householder_reflection(a)
        
        # Apply reflection to remaining matrix
        R[k:, k:] -= beta * np.outer(v, v.T @ R[k:, k:])
        
        # Accumulate transformations for Q
        Q[k:, :] -= beta * np.outer(v, v.T @ Q[k:, :])
    
    return Q.T, R  # Q.T because we accumulated transformations for Q^T

def solve_least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solves the least squares problem Ax ≈ b via QR decomposition.
    
    Parameters:
        A (np.ndarray): Coefficient matrix (m x n, m >= n)
        b (np.ndarray): Right-hand side vector (m,)
    
    Returns:
        np.ndarray: Solution vector x (n,)
    """
    Q, R = qr_decomposition(A)
    m, n = A.shape
    
    # Compute Q^T b
    Qt_b = Q.T @ b
    
    # Take upper part of R and corresponding elements of Q^T b
    R_upper = R[:n, :n]
    c = Qt_b[:n]
    
    # Solve upper triangular system
    x = np.linalg.solve(R_upper, c)
    return x

def example_usage():
    """
    Demonstration of the implemented functions
    """
    # Random matrix A and vector b
    m, n = 5, 3
    A = np.random.rand(m, n)
    b = np.random.rand(m)
    
    print("Matrix A:")
    print(A)
    print("\nVector b:")
    print(b)
    
    # Perform QR decomposition
    Q, R = qr_decomposition(A)
    print("\nMatrix Q:")
    print(Q)
    print("\nMatrix R:")
    print(R)
    
    # Verify decomposition correctness
    print("\nCheck QR = A:", np.allclose(Q @ R, A))
    print("Check Q^TQ=I:", np.allclose(Q.T @ Q, np.eye(Q.shape[0])))
    print("Check R is upper-triangular:", np.allclose(R, np.triu(R)))
    
    # Solve least squares problem
    x = solve_least_squares(A, b)
    print("\nLeast squares solution x:")
    print(x)
    
    # Check solution residual
    print("\nResidual ||Ax - b||:")
    print(np.linalg.norm(A @ x - b))
    
    # Compare with numpy.linalg.lstsq
    x_numpy, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    print("\nnumpy.linalg.lstsq solution:")
    print(x_numpy)
    print("\nDifference with performed solution:")
    print(np.linalg.norm(x - x_numpy))

if __name__ == "__main__":
    example_usage()