import numpy as np

def is_symmetric_tridiagonal(A, tol=1e-8):
    """
    Check if a matrix is symmetric and tridiagonal.
    
    Parameters:
        A (numpy.ndarray): Square matrix to check
        tol (float): Tolerance for symmetric and tridiagonal checks
        
    Returns:
        bool: True if matrix is symmetric tridiagonal, False otherwise
    """
    if not np.allclose(A, A.T, atol=tol):
        return False
    
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and abs(A[i, j]) > tol:
                return False
    return True

def ldl_tridiagonal_decomposition(A):
    """
    Perform LDLᵀ decomposition for symmetric tridiagonal matrix A = LDLᵀ.
    
    Parameters:
        A (numpy.ndarray): Symmetric tridiagonal matrix
        
    Returns:
        tuple: (L, D) where:
            L is unit lower bidiagonal matrix
            D is diagonal matrix
            
    Raises:
        np.linalg.LinAlgError: If matrix is not symmetric tridiagonal
    """
    if not is_symmetric_tridiagonal(A):
        raise np.linalg.LinAlgError("Matrix is not symmetric tridiagonal")
    
    n = A.shape[0]
    L = np.eye(n)  # Initialize as identity matrix
    D = np.zeros(n)  # Only store diagonal elements
    
    D[0] = A[0, 0]
    
    for i in range(1, n):
        L[i, i-1] = A[i, i-1] / D[i-1]
        D[i] = A[i, i] - L[i, i-1]**2 * D[i-1]
        # Early termination if matrix is not positive definite
        if D[i] <= 0:
            break
    
    return L, np.diag(D)

def solve_ldl_tridiagonal(L, D, b):
    """
    Solve linear system Ax = b using LDLᵀ decomposition.
    
    Parameters:
        L (numpy.ndarray): Unit lower bidiagonal matrix from LDLᵀ
        D (numpy.ndarray): Diagonal matrix from LDLᵀ
        b (numpy.ndarray): Right-hand side vector
        
    Returns:
        numpy.ndarray: Solution vector x
    """
    n = L.shape[0]
    
    # Forward substitution (solve Ly = b)
    y = np.zeros(n)
    y[0] = b[0]
    for i in range(1, n):
        y[i] = b[i] - L[i, i-1] * y[i-1]
    
    # Solve Dz = y
    z = y / np.diag(D)
    
    # Backward substitution (solve Lᵀx = z)
    x = np.zeros(n)
    x[-1] = z[-1]
    for i in range(n-2, -1, -1):
        x[i] = z[i] - L[i+1, i] * x[i+1]
    
    return x

def solve_tridiagonal_system(A, b):
    """
    Solve linear system Ax = b using LDLᵀ decomposition with verification.
    
    Parameters:
        A (numpy.ndarray): Symmetric tridiagonal matrix
        b (numpy.ndarray): Right-hand side vector
        
    Returns:
        tuple: (solution x, residual norm ||Ax - b||)
        
    Raises:
        np.linalg.LinAlgError: If matrix is not symmetric tridiagonal
                   If matrix is not positive definite
    """
    L, D = ldl_tridiagonal_decomposition(A)
    
    # Check for positive definiteness
    if np.any(np.diag(D) <= 0):
        raise np.linalg.LinAlgError("Matrix is not positive definite")
    
    x = solve_ldl_tridiagonal(L, D, b)
    residual = np.linalg.norm(A @ x - b)
    
    return x, residual

def generate_symmetric_tridiagonal(n, pos_def=True):
    """
    Generate a random symmetric tridiagonal matrix.
    
    Parameters:
        n (int): Size of the matrix
        pos_def (bool): Whether to generate positive definite matrix
        
    Returns:
        numpy.ndarray: n x n symmetric tridiagonal matrix
    """
    main_diag = np.random.rand(n) * 10 + (10 if pos_def else -5)
    off_diag = np.random.rand(n-1) * 2 - 1  # Values between -1 and 1
    
    A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    
    # Ensure positive definiteness if requested
    if pos_def:
        A = A @ A.T  # Makes diagonally dominant and positive definite
        # Restore tridiagonal structure
        A = np.diag(np.diag(A)) + np.diag(np.diag(A, k=1), k=1) + np.diag(np.diag(A, k=-1), k=-1)
    
    return A

def test_ldl_solver():
    """
    Test the LDLᵀ decomposition and solver with various matrices.
    """
    print("=== Testing LDLᵀ Decomposition ===")
    
    # Test with random positive definite matrix
    print("\nRandom positive definite tridiagonal matrix:")
    n = 5
    A = generate_symmetric_tridiagonal(n, pos_def=True)
    b = np.random.rand(n)
    
    print("A:\n", A)
    print("b:", b)
    
    try:
        x, residual = solve_tridiagonal_system(A, b)
        print("\nSolution x:", x)
        print("Residual ||Ax - b||:", residual)
        print("Verification A @ x - b:", A @ x - b)
        
        # Compare with numpy's solver
        x_np = np.linalg.solve(A, b)
        print("\nDifference from numpy solution:", np.linalg.norm(x - x_np))

    except np.linalg.LinAlgError as e:
        print("Error:", e)
    
    # Test with non-positive definite matrix
    print("\nRandom non-positive definite tridiagonal matrix:")
    A = generate_symmetric_tridiagonal(n, pos_def=False)
    b = np.random.rand(n)
    
    print("A:\n", A)
    print("b:", b)
    
    try:
        x, residual = solve_tridiagonal_system(A, b)
        print("\nSolution x:", x)

    except np.linalg.LinAlgError as e:
        print("Error:", e)

if __name__ == "__main__":
    test_ldl_solver()