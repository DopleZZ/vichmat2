import numpy
from dto.equation import Equation
from dto.result import Result
from methods.method import Method


dx = 0.00001
steps = 100
MAX_ITERS = 50_000

class SimpleIterationsMethod(Method):
    name = 'Метод простой итерации'

    def __init__(self, equation: Equation, left: float, right: float,
                 epsilon: float, decimal_places: int, log: bool):
        super().__init__(equation, left, right, epsilon, decimal_places, log)

    def check(self):
        if not self.equation.root_exists(self.left, self.right):
            return False, 'Отсутствует корень на заданном промежутке или корней > 2'
        return True, ''

    def solve(self) -> Result:
        f = self.equation.function
        # Начальное приближение
        x = (self.left + self.right) / 2

        # Подбор параметра lbd на основе максимума производной
        der_left = abs(self.equation.derivative(self.left))
        der_right = abs(self.equation.derivative(self.right))
        max_derivative = max(der_left, der_right)
        if max_derivative == 0:
            raise ValueError('Производная равна нулю на концах интервала, метод простой итерации невозможен')
        lbd = 1 / max_derivative
        # Построение функции итераций
        phi = lambda t: t + lbd * f(t)

        # Проверка условия сходимости |phi'(x)| < 1 на отрезке
        for xi in numpy.linspace(self.left, self.right, steps):
            phi_der = abs(1 + lbd * self.equation.derivative(xi))
            if phi_der >= 1:
                raise ValueError(f"Не выполнено условие сходимости: |phi'(x)| = {phi_der:.3f} >= 1 при x = {xi:.3f}")

        iteration = 0
        last_x = x
        while True:
            iteration += 1
            if iteration > MAX_ITERS:
                raise ValueError(f"Превышено максимальное число итераций ({MAX_ITERS}) без сходимости")

            x = phi(last_x)
           

            # Условие остановки
            if abs(x - last_x) <= self.epsilon and abs(f(x)) <= self.epsilon:
                break

            last_x = x

        return Result(x, f(x), iteration, self.decimal_places)
