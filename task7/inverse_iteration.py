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

# Использование для комплексной матрицы
if __name__ == "__main__":

    A_complex = np.array([[3+2j, 1, 0], [0, 5-1j, 2], [0, 0, 2+3j]], dtype=complex)
    mu_complex = 2.1 + 3.1j  # Близко к lambda = 2+3j
    vec, val = inverse_power_iteration(A_complex, mu_complex)
    print("\nСобственный вектор (комплексный случай):\n", vec)
    print("Соответствующее собственное значение:", val)