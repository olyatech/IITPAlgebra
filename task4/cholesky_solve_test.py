import numpy as np
import pytest
from cholesky import solve_cholesky

def test_solve_big_random():
    matrix_size = 100

    A = np.random.rand(matrix_size, matrix_size)
    A = A.T @ A + np.eye(matrix_size) * 0.1

    b = np.random.rand(matrix_size)

    while np.linalg.matrix_rank(A) != matrix_size:
        A = np.random.rand(matrix_size, matrix_size) 
        A = A.T @ A + np.eye(matrix_size) * 0.1

    x = solve_cholesky(A, b)
    expected_x = np.linalg.solve(A, b)
    assert np.allclose(x, expected_x), "Comparing with numpy solve failed"

def test_negative_definite():
    matrix_size = 100
    A = np.random.rand(matrix_size, matrix_size)
    A = A.T @ A + np.eye(matrix_size) * 0.1

    b = np.random.rand(matrix_size)

    with pytest.raises(np.linalg.LinAlgError):
        solve_cholesky(-A, b)

def test_solve_singular():
    matrix_size = 100

    A = np.random.rand(matrix_size, matrix_size)
    b = np.random.rand(matrix_size + 1)

    A[matrix_size - 1] = np.zeros(matrix_size)

    with pytest.raises(np.linalg.LinAlgError):
        solve_cholesky(A, b)
    

def test_solve_dimension_mismatch():
    matrix_size = 100

    A = np.random.rand(matrix_size, matrix_size)
    b = np.random.rand(matrix_size + 1)

    with pytest.raises(np.linalg.LinAlgError):
        solve_cholesky(A, b)

def test_solve_non_square():
    matrix_size = 100

    A = np.random.rand(matrix_size, matrix_size + 1)
    b = np.random.rand(matrix_size )

    with pytest.raises(np.linalg.LinAlgError):
        solve_cholesky(A, b)

# Run the tests
if __name__ == "__main__":
    pytest.main()