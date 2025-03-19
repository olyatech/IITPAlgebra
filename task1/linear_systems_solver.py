import numpy as np

def lu_decomposition(A):
    """
    Perform LU decomposition with partial pivoting: PA = LU.

    Parameters:
    A (np.array): Coefficient matrix.

    Returns:
    P (np.array): Permutation matrix.
    L (np.array): Lower triangular matrix.
    U (np.array): Upper triangular matrix.
    """
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n):
        # Partial pivoting: choose the best column and swap
        max_index = np.argmax(np.abs(U[k:, k])) + k
        if k != max_index:
            U[[k, max_index]] = U[[max_index, k]]
            P[[k, max_index]] = P[[max_index, k]]
            if k > 0:
                L[[k, max_index], :k] = L[[max_index, k], :k]

        # Elimination
        for j in range(k + 1, n):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k:] -= L[j, k] * U[k, k:]

    return P, L, U

def forward_pass(L, P, b):
    """
    Solve Ly = Pb using forward substitution.

    Parameters:
    L (np.array): Lower triangular matrix.
    P (np.array): Permutation matrix.
    b (np.array): Right-hand side vector.

    Returns:
    y (np.array): Solution vector.
    """
    if L.shape[0] != b.shape[0]:
        raise np.linalg.LinAlgError("Matrix L and vector b dimension mismatch")
    
    Pb = np.dot(P, b)
    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])

    return y

def backward_pass(U, y):
    """
    Solve Ux = y using back substitution.

    Parameters:
    U (np.array): Upper triangular matrix.
    b (np.array): Right-hand side vector.

    Returns:
    x (np.array): Solution vector.
    """
    if U.shape[0] != y.shape[0]:
        raise np.linalg.LinAlgError("Matrix U and vector y dimension mismatch")
    
    x = np.zeros_like(y)
    for i in range(len(y) - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def solve_lu(A, b):
    """
    Solve the system Ax = b using LU decomposition with partial pivoting.

    Parameters:
    A (np.array): Square coefficient matrix.
    b (np.array): Right-hand side vector.

    Returns:
    x (np.array): Solution vector.
    P (np.array): Permutation matrix.
    L (np.array): Lower triangular matrix.
    U (np.array): Upper triangular matrix.
    """

    if A.shape[0] != A.shape[1]:
        raise np.linalg.LinAlgError("Matrix A is not square")

    if A.shape[0] != b.shape[0]:
        raise np.linalg.LinAlgError("Matrix A and vector b dimension mismatch")

    if A.shape[0] != np.linalg.matrix_rank(A):
        raise np.linalg.LinAlgError("Singular matrix A was given")
    
    P, L, U = lu_decomposition(A)

    # Solve Ly = Pb using forward substitution
    y = forward_pass(L, P, b)

    # Solve Ux = y using back substitution
    x = backward_pass(U, y)

    return x, P, L, U

def example_usage():
    # Simple test
    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)

    print("Try simple test with small matrices.")
    print("Matrix A:\n", A)
    print("Vector b:\n", b)


    # Show LU decomposition
    solution, P, L, U = solve_lu(A, b)

    print("Permutation Matrix P:\n", P)
    print("Lower Triangular Matrix L:\n", L)
    print("Upper Triangular Matrix U:\n", U)
    print("Solution:", solution)

    # Show difference with numpy.linalg.solve
    expected_solution = np.linalg.solve(A, b)
    norm_difference = np.linalg.norm(solution - expected_solution)
    print("For small simple test norm of difference with numpy.linalg.solve:", norm_difference)

    # try big random matrices
    matrix_size = 100

    A = np.random.rand(matrix_size, matrix_size)
    b = np.random.rand(matrix_size)
    while np.linalg.matrix_rank(A) != matrix_size:
       A = np.random.rand(matrix_size, matrix_size) 

    solution, P, L, U = solve_lu(A, b)
    expected_solution = np.linalg.solve(A, b)

    norm_difference = np.linalg.norm(solution - expected_solution)
    print("For random matrix 100x100 norm of difference with numpy.linalg.solve:", norm_difference)

if __name__ == "__main__":
    example_usage()