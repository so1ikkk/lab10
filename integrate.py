import math
import unittest
from typing import Callable


def integrate(f: Callable[[float], float], a: float, b: float, *, n_iter: int = 100_000) -> float:
    """
    Вычисляет определённый интеграл функции `f` на интервале [a, b] методом прямоугольников.

    Метод основан на разбиении интервала [a, b] на `n_iter` равных частей и
    суммировании площадей прямоугольников, построенных на значениях функции
    в левом конце каждого подинтервала.

    Args:
        f (Callable[[float], float]): Функция одного аргумента, которую нужно интегрировать.
        a (float): Левая граница интегрирования.
        b (float): Правая граница интегрирования.
        n_iter (int, optional): Количество шагов разбиения интервала.
                                Чем больше значение, тем выше точность.
                                По умолчанию 100_000.

    Returns:
        float: Приближённое значение определённого интеграла функции `f` на [a, b].

    Raises:
        ValueError: Если n_iter <= 0 или a >= b.

    Examples:
        >>> round(integrate(math.cos, 0, math.pi, n_iter=1000),2)
        0.0
        >>> round(integrate(lambda x: x**2, 0, 1, n_iter=1000),2)
        0.33
    """
    if n_iter <= 0:
        raise ValueError("n_iter должно быть положительным")
    if a >= b:
        raise ValueError("Левая граница a должна быть меньше правой b")

    acc = 0.0
    step = (b - a) / n_iter
    for i in range(n_iter):
        acc += f(a + i * step) * step
    return acc


class TestIntegrate(unittest.TestCase):
    def test_trig_function(self):
        # интеграл cos(x) от 0 до pi равен 0
        result = integrate(math.cos, 0, math.pi, n_iter=1_000_000)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_polynomial_function(self):
        # интеграл x^2 от 0 до 1 равен 1/3
        result = integrate(lambda x: x**2, 0, 1, n_iter=1_000_000)
        self.assertAlmostEqual(result, 1/3, places=5)

    def test_n_iter_effect(self):
        # Проверка устойчивости к различному количеству итераций
        res_low = integrate(lambda x: x**2, 0, 1, n_iter=100)
        res_high = integrate(lambda x: x**2, 0, 1, n_iter=1_000_000)
        self.assertTrue(abs(res_high - res_low) > 1e-5)

    def test_invalid_parameters(self):
        # Проверка обработки ошибок
        with self.assertRaises(ValueError):
            integrate(lambda x: x**2, 1, 0)
        with self.assertRaises(ValueError):
            integrate(lambda x: x**2, 0, 1, n_iter=0)

if __name__ == "__main__":
    unittest.main()

