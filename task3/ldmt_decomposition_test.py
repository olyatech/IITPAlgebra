import numpy as np
import pytest

from ldmt_decomposition import ldmt_decomposition

def test_lower_triangular():
    matrix_size = 100
    A = np.random.rand(matrix_size, matrix_size)
    while np.linalg.matrix_rank(A) != matrix_size:
       A = np.random.rand(matrix_size, matrix_size) 

    L, D, M = ldmt_decomposition(A)
    assert np.allclose(L, np.tril(L)), "L is not lower triangular"

def test_diagonal():
    matrix_size = 100
    A = np.random.rand(matrix_size, matrix_size)
    while np.linalg.matrix_rank(A) != matrix_size:
       A = np.random.rand(matrix_size, matrix_size) 

    L, D, M = ldmt_decomposition(A)
    assert np.allclose(D, np.diag(np.diag(D))), "D is not diagonal"

def test_upper_triangular():
    matrix_size = 100
    A = np.random.rand(matrix_size, matrix_size)
    while np.linalg.matrix_rank(A) != matrix_size:
       A = np.random.rand(matrix_size, matrix_size) 

    L, D, M = ldmt_decomposition(A)
    assert np.allclose(M, np.tril(M)), "M^T is not upper triangular"

def test_allclose():
    matrix_size = 100
    A = np.random.rand(matrix_size, matrix_size)
    while np.linalg.matrix_rank(A) != matrix_size:
       A = np.random.rand(matrix_size, matrix_size) 

    L, D, M = ldmt_decomposition(A)
    assert np.allclose(A, L @ D @ M.T), "A is not equal to L * D * M^T"

def test_non_square():
    matrix_size_1 = np.random.randint(1, 100)
    matrix_size_2 = np.random.randint(1, 100)

    while matrix_size_1 == matrix_size_2:
        matrix_size_2 = np.random.randint(1, 100)

    A = np.random.rand(matrix_size_1, matrix_size_2)

    with pytest.raises(np.linalg.LinAlgError):
        ldmt_decomposition(A)

def test_singular():
    A = np.array([[1, 2], [2, 4]], dtype=float)

    with pytest.raises(np.linalg.LinAlgError):
        ldmt_decomposition(A)

# Run the tests
if __name__ == "__main__":
    pytest.main()