import numpy as np
import pytest

from ldmt_decomposition import is_positive_definite

def test_positive_definite():
    matrix_size = 100
    A = np.random.rand(matrix_size, matrix_size)
    A = A.T @ A + np.eye(matrix_size) * 0.1

    assert (is_positive_definite(A) == True), "A is positive definite"

def test_negative_definite():
    matrix_size = 100
    A = np.random.rand(matrix_size, matrix_size)
    A = A.T @ A + np.eye(matrix_size) * 0.1

    assert (is_positive_definite(-A) == False), "A is negative definite"

def test_non_square():
    matrix_size_1 = np.random.randint(1, 100)
    matrix_size_2 = np.random.randint(1, 100)

    while matrix_size_1 == matrix_size_2:
        matrix_size_2 = np.random.randint(1, 100)

    A = np.random.rand(matrix_size_1, matrix_size_2)

    with pytest.raises(np.linalg.LinAlgError):
        is_positive_definite(A)

def test_singular():
    A = np.array([[1, 2], [2, 4]], dtype=float)

    with pytest.raises(np.linalg.LinAlgError):
        is_positive_definite(A)

# Run the tests
if __name__ == "__main__":
    pytest.main()