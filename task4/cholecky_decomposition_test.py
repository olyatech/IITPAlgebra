import numpy as np
import pytest
from cholesky import cholesky_decomposition

def test_solve_big_random():
    matrix_size = 100

    A = np.random.rand(matrix_size, matrix_size)
    A = A.T @ A + np.eye(matrix_size) * 0.1

    while np.linalg.matrix_rank(A) != matrix_size:
        A = np.random.rand(matrix_size, matrix_size) 
        A = A.T @ A + np.eye(matrix_size) * 0.1

    L = cholesky_decomposition(A)
    assert np.allclose(np.dot(L, L.T), A), "Cholesky decomposition failed"

def test_negative_definite():
    matrix_size = 100
    A = np.random.rand(matrix_size, matrix_size)
    A = A.T @ A + np.eye(matrix_size) * 0.1

    with pytest.raises(np.linalg.LinAlgError):
        cholesky_decomposition(-A)

def test_solve_singular():
    matrix_size = 100

    A = np.random.rand(matrix_size, matrix_size)
    A[matrix_size - 1] = np.zeros(matrix_size)

    with pytest.raises(np.linalg.LinAlgError):
        cholesky_decomposition(A)
    

def test_solve_not_symmetric():
    matrix_size = 100
    A = np.random.rand(matrix_size, matrix_size)
    A = A.T @ A + np.eye(matrix_size) * 0.1
    A[0] += 1

    with pytest.raises(np.linalg.LinAlgError):
        cholesky_decomposition(A)

def test_solve_non_square():
    matrix_size = 100
    A = np.random.rand(matrix_size, matrix_size + 1)

    with pytest.raises(np.linalg.LinAlgError):
        cholesky_decomposition(A)

# Run the tests
if __name__ == "__main__":
    pytest.main()