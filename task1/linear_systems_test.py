import numpy as np
import pytest
from linear_systems_solver import solve_lu 

def test_solve_lu_basic():
    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)
    solution, P, L, U = solve_lu(A, b)
    expected_solution = np.linalg.solve(A, b)
    assert np.allclose(solution, expected_solution), "Test failed for basic case"

def test_solve_lu_identity():
    A = np.eye(3)
    b = np.array([1, 2, 3], dtype=float)
    solution, P, L, U = solve_lu(A, b)
    expected_solution = np.linalg.solve(A, b)
    assert np.allclose(solution, expected_solution), "Test failed for identity matrix"

def test_solve_lu_big_random():
    matrix_size = 100

    A = np.random.rand(matrix_size, matrix_size)
    b = np.random.rand(matrix_size)

    while np.linalg.matrix_rank(A) != matrix_size:
       A = np.random.rand(matrix_size, matrix_size) 

    solution, P, L, U = solve_lu(A, b)
    expected_solution = np.linalg.solve(A, b)
    assert np.allclose(solution, expected_solution), "Test failed for random matrix"

def test_solve_lu_singular():
    A = np.array([[1, 2], [2, 4]], dtype=float)
    b = np.array([1, 2], dtype=float)
    with pytest.raises(np.linalg.LinAlgError):
        solve_lu(A, b)

def test_solve_lu_dimension_mismatch():
    A = np.array([[2, 1, -1], [-3, -1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)
    with pytest.raises(np.linalg.LinAlgError):
        solve_lu(A, b)

def test_solve_lu_non_square():
    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2], [1, 0, 0]], dtype=float)
    b = np.array([8, -11, -3, 0], dtype=float)
    with pytest.raises(np.linalg.LinAlgError):
        solve_lu(A, b)

# Run the tests
if __name__ == "__main__":
    pytest.main()