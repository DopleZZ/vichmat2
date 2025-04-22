import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

# Импорты из проекта
from methods.half_division_method import HalfDivisionMethod
from methods.chord_method import ChordMethod
from methods.simple_iterations_method import SimpleIterationsMethod
from methods.newton_method import NewtonMethod
from dto.equation import Equation
import system_of_equation

# Предопределенные функции (из main.py)
predefined_functions = {
    1: Equation(
        lambda x: -1.38*x**3 - 5.42*x**2 + 2.57*x + 10.95,
        '-1.38*x^3 - 5.42*x^2 + 2.57*x + 10.95',
        lambda x: -1.38*3*x**2 - 5.42*2*x + 2.57
    ),
    2: Equation(
        lambda x: x**3 - 1.89*x**2 - 2*x + 1.76,
        'x^3 - 1.89*x^2 - 2*x + 1.76',
        lambda x: 3*x**2 - 2*1.89*x - 2
    ),
    3: Equation(
        lambda x: x / 2 - 2 * (x + 2.39)**(1 / 3),
        'x/2 - 2*(x + 2.39)^(1/3)',
        lambda x: 0.5 - (2 / 3) * (x + 2.39) ** (-2 / 3)
    ),
    4: Equation(
        lambda x: -x / 2 + math.e ** x + 5 * math.sin(x),
        '-x/2 + e^x + 5*sin(x)',
        lambda x: -0.5 + math.e**x + 5 * math.cos(x)
    )
}

methods = {
    "Метод половинного деления": HalfDivisionMethod,
    "Метод хорд": ChordMethod,
    "Метод простой итерации": SimpleIterationsMethod,
    "Метод Ньютона": NewtonMethod,
}

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Численные методы: GUI")
        self.geometry("900x700")

        # Режим: уравнение или система
        self.mode_var = tk.StringVar(value="equation")
        modes = [("Уравнение", "equation"), ("Система уравнений", "system")]
        for text, mode in modes:
            rb = ttk.Radiobutton(self, text=text, variable=self.mode_var, value=mode, command=self.on_mode_change)
            rb.pack(anchor=tk.W, padx=10)

        # Frame для уравнений
        self.eq_frame = ttk.Frame(self)
        self.eq_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.eq_frame, text="Выберите уравнение:").grid(row=0, column=0, padx=5, pady=5)
        eq_texts = [f"{i}: {eq.text}" for i, eq in predefined_functions.items()]
        self.eq_combo = ttk.Combobox(self.eq_frame, values=eq_texts, state="readonly")
        self.eq_combo.grid(row=0, column=1, padx=5, pady=5)
        self.eq_combo.bind("<<ComboboxSelected>>", lambda e: self.enable_solve())

        # Frame для методов и параметров уравнения
        self.method_frame = ttk.Frame(self)
        self.method_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.method_frame, text="Выберите метод:").grid(row=0, column=0, padx=5, pady=5)
        self.method_combo = ttk.Combobox(self.method_frame, values=list(methods.keys()), state="readonly")
        self.method_combo.grid(row=0, column=1, padx=5, pady=5)
        self.method_combo.bind("<<ComboboxSelected>>", lambda e: self.on_method_selected())

        # Общий Frame для ввода параметров
        self.input_frame = ttk.Frame(self)
        self.input_frame.pack(fill=tk.X, pady=5)
        # Погрешность
        self.eps_label = ttk.Label(self.input_frame, text="Погрешность:")
        self.eps_entry = ttk.Entry(self.input_frame)
        # Интервал
        self.left_label = ttk.Label(self.input_frame, text="Левая граница:")
        self.left_entry = ttk.Entry(self.input_frame)
        self.right_label = ttk.Label(self.input_frame, text="Правая граница:")
        self.right_entry = ttk.Entry(self.input_frame)
        # Начальное приближение для Ньютона
        self.x0_label = ttk.Label(self.input_frame, text="Начальное приближение x0:")
        self.x0_entry = ttk.Entry(self.input_frame)

        # Кнопка Решить
        self.solve_btn = ttk.Button(self, text="Решить", command=self.solve)
        self.solve_btn.pack(pady=5)
        self.solve_btn.config(state=tk.DISABLED)

        # Результат
        self.result_text = tk.Text(self, height=5, wrap="word")
        self.result_text.pack(fill=tk.X, padx=10, pady=5)
        self.result_text.config(state=tk.DISABLED)

        # Полотно для графика
        self.fig, self.ax = plt.subplots(figsize=(6,4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Frame для системы
        self.sys_frame = ttk.Frame(self)
        ttk.Label(self.sys_frame, text="Выберите систему:").grid(row=0, column=0, padx=5, pady=5)
        self.sys_combo = ttk.Combobox(self.sys_frame, values=["1: x^2+y^2-1, x^2-y-0.5"], state="readonly")
        self.sys_combo.grid(row=0, column=1, padx=5, pady=5)
        self.sys_combo.bind("<<ComboboxSelected>>", lambda e: self.enable_solve())
        ttk.Label(self.sys_frame, text="x0:").grid(row=1, column=0, padx=5, pady=5)
        self.x0_sys = ttk.Entry(self.sys_frame)
        self.x0_sys.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(self.sys_frame, text="y0:").grid(row=2, column=0, padx=5, pady=5)
        self.y0_sys = ttk.Entry(self.sys_frame)
        self.y0_sys.grid(row=2, column=1, padx=5, pady=5)
        ttk.Label(self.sys_frame, text="Погрешность:").grid(row=3, column=0, padx=5, pady=5)
        self.eps_sys = ttk.Entry(self.sys_frame)
        self.eps_sys.grid(row=3, column=1, padx=5, pady=5)

        self.on_mode_change()

    def on_mode_change(self):
        mode = self.mode_var.get()
        if mode == "equation":
            self.eq_frame.pack(fill=tk.X, pady=5)
            self.method_frame.pack(fill=tk.X, pady=5)
            self.input_frame.pack(fill=tk.X, pady=5)
            self.sys_frame.pack_forget()
        else:
            self.eq_frame.pack_forget()
            self.method_frame.pack_forget()
            self.input_frame.pack_forget()
            self.sys_frame.pack(fill=tk.X, pady=5)
        self.enable_solve()

    def on_method_selected(self):
        for w in self.input_frame.winfo_children():
            w.grid_forget()
        self.eps_label.grid(row=0, column=0, padx=5, pady=5)
        self.eps_entry.grid(row=0, column=1, padx=5, pady=5)
        method = self.method_combo.get()
        if method == "Метод Ньютона":
            self.x0_label.grid(row=1, column=0, padx=5, pady=5)
            self.x0_entry.grid(row=1, column=1, padx=5, pady=5)
        else:
            self.left_label.grid(row=1, column=0, padx=5, pady=5)
            self.left_entry.grid(row=1, column=1, padx=5, pady=5)
            self.right_label.grid(row=2, column=0, padx=5, pady=5)
            self.right_entry.grid(row=2, column=1, padx=5, pady=5)
        self.enable_solve()

    def enable_solve(self):
        mode = self.mode_var.get()
        if mode == "equation":
            if self.eq_combo.get() and self.method_combo.get():
                self.solve_btn.config(state=tk.NORMAL)
            else:
                self.solve_btn.config(state=tk.DISABLED)
        else:
            if self.sys_combo.get():
                self.solve_btn.config(state=tk.NORMAL)
            else:
                self.solve_btn.config(state=tk.DISABLED)

    def solve(self):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.ax.clear()
        try:
            if self.mode_var.get() == "equation":
                idx = int(self.eq_combo.get().split(":")[0])
                eq = predefined_functions[idx]
                method_name = self.method_combo.get()
                MethodClass = methods[method_name]
                eps = float(self.eps_entry.get().replace(',', '.'))
                dec = abs(len(str(eps).split('.')[-1])) if '.' in str(eps) else 0
                if MethodClass is NewtonMethod:
                    x0 = float(self.x0_entry.get().replace(',', '.'))
                    method = MethodClass(eq, x0, 0, eps, dec, log=False)
                    a, b = x0-5, x0+5
                else:
                    a = float(self.left_entry.get().replace(',', '.'))
                    b = float(self.right_entry.get().replace(',', '.'))
                    method = MethodClass(eq, a, b, eps, dec, log=False)
                ok, msg = method.check()
                if not ok:
                    raise ValueError(msg)
                res = method.solve()
                self.result_text.insert(tk.END, str(res))
                x = np.linspace(a, b, 400)
                y = np.vectorize(eq.function)(x)
                self.ax.plot(x, y, label=eq.text)
                self.ax.axhline(0, color='gray', linewidth=0.8)
                root = res.root
                self.ax.plot(root, eq.function(root), 'ro', label=f'root={root:.{dec}f}')
                self.ax.legend()
            else:
                eps = float(self.eps_sys.get())
                x0 = float(self.x0_sys.get())
                y0 = float(self.y0_sys.get())
                a_fun = system_of_equation.a
                sol, iters = system_of_equation.solve(a_fun, system_of_equation.phi1, system_of_equation.phi2, (x0,y0), eps)
                if sol is None:
                    raise ValueError("Не сошлось или ошибка в методе.")
                self.result_text.insert(tk.END, f"x = {sol[0]:.5f}, y = {sol[1]:.5f}\nИтераций: {iters}")
                x = np.linspace(x0-5, x0+5, 200)
                y = np.linspace(y0-5, y0+5, 200)
                X, Y = np.meshgrid(x, y)
                Z1 = np.vectorize(lambda xx, yy: a_fun((xx,yy))[0])(X, Y)
                Z2 = np.vectorize(lambda xx, yy: a_fun((xx,yy))[1])(X, Y)
                self.ax.contour(X, Y, Z1, levels=[0], colors='r')
                self.ax.contour(X, Y, Z2, levels=[0], colors='b')
                self.ax.plot(sol[0], sol[1], 'go', label='solution')
                self.ax.legend()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
        self.canvas.draw()
        self.result_text.config(state=tk.DISABLED)

if __name__ == '__main__':
    app = Application()
    app.mainloop()
