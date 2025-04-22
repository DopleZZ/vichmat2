from typing import Callable
import matplotlib.pyplot as plt
import numpy as np


dx = 0.00001

class Equation:
    def __init__(self, function: Callable, text: str, derivative: Callable):
        self.text = text
        self.function = function
        self.derivative = derivative

    def draw(self, left: float, right: float):
        x = np.linspace(left, right)
        func = np.vectorize(self.function)(x)

        plt.title = 'График заданной функции'
        plt.grid(True, which='both')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axhline(y=0, color='gray', label='y = 0')
        plt.plot(x, func, 'blue', label=self.text)
        plt.legend(loc='upper left')
        plt.savefig('graph.png')
        plt.show()

    def root_exists(self, left: float, right: float):
        return (self.function(left) * self.function(right) < 0) \
               and (self.derivative(left) * self.derivative(left) > 0)