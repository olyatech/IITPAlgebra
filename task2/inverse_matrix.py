import numpy as np
import sys

sys.path.append("..")

from task1.linear_systems_solver import lu_decomposition, forward_pass, backward_pass

def matrix_inverse(A):
    """
    Calculate the inverse of matrix A using LU decomposition.

    Parameters:
    A (numpy array): Input matrix.

    Returns:
    A_inv (numpy array): Inverse of matrix A.
    """
    if A.shape[0] != A.shape[1]:
        raise np.linalg.LinAlgError("Matrix A is not square")
    
    if A.shape[0] != np.linalg.matrix_rank(A):
        raise np.linalg.LinAlgError("Singular matrix A was given")
    
    n = A.shape[0]
    P, L, U = lu_decomposition(A)
    A_inv = np.zeros((n, n))

    # Solve for each column of the inverse
    for i in range(n):
        b = np.zeros(n)
        b[i] = 1

        y = forward_pass(L, P, b)
        x = backward_pass(U, y)
        A_inv[:, i] = x

    return A_inv

def example_usage():
    """
    Creates random matrix 100x100, calculates A^{-1} and prints the norm of (AA^{-1} - I)
    """
    matrix_size = 100

    A = np.random.rand(matrix_size, matrix_size)
    A_inv = matrix_inverse(A)

    norm_difference = np.linalg.norm(np.dot(A, A_inv) - np.eye(matrix_size))
    print("Norm of difference with numpy.linalg.inv:", norm_difference)

if __name__ == "__main__":
    example_usage()