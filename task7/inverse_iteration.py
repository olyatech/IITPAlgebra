import numpy as np
from numpy.linalg import norm, solve
from typing import Union, Optional

def inverse_power_iteration(
    A: np.ndarray,
    mu: Union[float, complex],
    max_iter: int = 100,
    tol: float = 1e-8,
    x0: Optional[np.ndarray] = None
) -> tuple[np.ndarray, Union[float, complex]]:
    """
    Алгоритм обратных степенных итераций со сдвигом для поиска собственного вектора и значения.

    Параметры:
        A (np.ndarray): Квадратная матрица размера (n, n).
        mu (Union[float, complex]): Приближение к целевому собственному значению.
        max_iter (int): Максимальное число итераций. По умолчанию 100.
        tol (float): Критерий остановки (норма разности векторов). По умолчанию 1e-8.
        x0 (Optional[np.ndarray]): Начальное приближение вектора. Если None, генерируется случайно.

    Возвращает:
        tuple[np.ndarray, Union[float, complex]]: Приближенный собственный вектор и соответствующее значение.
    """
    n = A.shape[0]
    I = np.eye(n, dtype=A.dtype)
    A_shift = A - mu * I  # Матрица со сдвигом
    
    # Инициализация начального вектора
    x = x0 if x0 is not None else np.random.rand(n)
    x = x.astype(A.dtype)  # Сохранение типа данных (комплексный/вещественный)
    x /= norm(x)  # Нормировка
    
    for _ in range(max_iter):
        x_prev = x.copy()
        
        # Решение системы (A - mu I)x_new = x_prev
        try:
            x = solve(A_shift, x_prev)  # LU-разложение для общего случая
        except np.linalg.LinAlgError:
            raise ValueError("Матрица (A - mu I) вырождена")
        
        # Нормировка вектора
        x /= norm(x)
        
        # Проверка сходимости
        if norm(x - x_prev) < tol:
            break
    
    # Вычисление соответствующего собственного значения через Рэлеевское отношение
    eigenvalue = (x.conj().T @ A @ x) / (x.conj().T @ x)
    return x, eigenvalue

import numpy as np
from numpy.linalg import norm

def test_complex_diagonal_matrix():
    """
    Тест для случайной верхней треугольной матрицы с комплексными диагональными элементами.
    Проверяет, что алгоритм обратных степенных итераций находит собственный вектор и значение,
    близкие к выбранному диагональному элементу.
    """
    np.random.seed(42)  # Для воспроизводимости
    np.set_printoptions(precision=2)
    
    # Генерация случайной верхней треугольной матрицы nxn
    n = 3
    A = np.zeros((n, n), dtype=complex)
    
    diag = 1 + 4 * (np.random.rand(n) + 1j * np.random.rand(n))
    np.fill_diagonal(A, diag)
    
    for i in range(n):
        for j in range(i+1, n):
            A[i, j] = np.random.rand()
    
    # Выбор целевого собственного значения (ближайшего к mu)
    target_idx = 1  # Индекс диагонального элемента
    target_eigenvalue = diag[target_idx]
    mu = target_eigenvalue + 0.01 * (1 + 1j)  # Небольшой сдвиг
    
    eigenvector, computed_eigenvalue = inverse_power_iteration(A, mu, tol=1e-10)
    
    # Сравниваем вычисленное и целевое собственные значения
    error_eigenvalue = abs(computed_eigenvalue - target_eigenvalue)
    print(f"Вычисленное собственное значение: {computed_eigenvalue:.6f}")
    print(f"Ожидаемое собственное значение:   {target_eigenvalue:.6f}")
    print(f"Ошибка: {error_eigenvalue:.6e}")
    assert error_eigenvalue < 1e-6, "Собственное значение не совпадает с ожидаемым!"
    
    # Проверка, что вектор близок к каноническому базисному вектору
    expected_vector = np.zeros(n, dtype=complex)
    expected_vector[target_idx] = 1.0
    print(f"Найденный вектор: {eigenvector}")
    print(f"Ожидаемый вектор: {expected_vector}")

    if abs(abs(np.dot(expected_vector, eigenvector)) - 1) < 1e-2:
        print(f"Полученный вектор близок к коллинеарному с ожидаемым, скалярное произведение {np.dot(expected_vector, eigenvector)}")


# Использование для комплексной матрицы
if __name__ == "__main__":
    test_complex_diagonal_matrix()

