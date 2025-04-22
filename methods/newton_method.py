import numpy as np
from dto.result import Result
from methods.method import Method


dx = 0.00001

class NewtonMethod(Method):
    name = 'Метод Ньютона'

    def solve(self) -> Result:
        f = self.equation.function
        x0 = self.left

        epsilon = self.epsilon
        iteration = 0

        while True:
            iteration += 1

            df = self.equation.derivative(x0)
            x1 = x0 - f(x0) / df

            if abs(x1 - x0) < epsilon and f(x1) < epsilon:
                break

            x0 = x1

        return Result(x1, f(x1), iteration, self.decimal_places)