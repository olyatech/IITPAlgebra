# Задания по курсу прикладной линейной алгебры в ИППИ РАН

## Задача 1
Реализовать разложение Гаусса $PA = LU$ (с выбором ведущего  элемента), использовать его для решения линейной системы $Ax = b$. 
Организовать проверку, вычислив $A\hat{x} - b$, где $\hat{x}$ -- найденное решение.

Реализация лежит в файле `task1/linear_systems_solver.py`, если запустить его, то выдаст разложение на примере матрицы $3\times 3$, и запустит тест для рандомной матрицы $100\times 100$. 

В файле `task1/linear_systems_test.py` лежат pytest'ы для функции решения уравнений, их тоже можно запустить.