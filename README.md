# Задания по курсу прикладной линейной алгебры в ИППИ РАН

## Задача 1
Реализовать разложение Гаусса $PA = LU$ (с выбором ведущего  элемента), использовать его для решения линейной системы $Ax = b$. 
Организовать проверку, вычислив $Ax_0 - b$, где $x_0$ найденное решение.

Реализация лежит в файле `task1/linear_systems_solver.py`, если запустить его, то выдаст разложение на примере матрицы $3\times 3$, и запустит тест для рандомной матрицы $100\times 100$. 

В файле `task1/linear_systems_test.py` лежат pytest'ы для функции решения уравнений, их тоже можно запустить.

## Задача 2
Реализовать разложение Гаусса $PA = LU$ (с выбором ведущего  элемента), использовать его для нахождения обратной матрицы $A^{-1}$. 
Организовать проверку, вычислив $A\hat{A^{-1}}$, где $\hat{A^{-1}}$ -- найденное решение.

Реализация лежит в файле `task2/inverse_matrix.py`, если запустить его, то будет запущен поиск обратной матрицы для случайной матрицы $100\times 100$, будет выведена норма разницы $A\hat{A^{-1}}$ и единичной матрицы. 
