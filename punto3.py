"""Punto 3: elementos finitos P3 (Hermite) para u''' - u'' = f en (0,1),
u(0) = u(1) = 0, u'(0) = u'(1) = 0.

Formulación variacional: hallar u en H^2_0 con  a(u,v) = -int u'' v' + int u' v' = int f v.
Solución exacta propuesta: la misma del punto 2, u(x) = x^2 (1-x)^2 e^x
->  f(x) = (4x^3 + 18x^2 + 2x - 8) e^x.
"""
import numpy as np
from punto2 import (uex, f as f2, bloques, matriz_tercer_orden, estudio, figura_solucion)


def f(x):
    return (4 * x**3 + 18 * x**2 + 2 * x - 8) * np.exp(x)


def matriz_rigidez_K1(n):
    """Matriz de int u' v' con la base de Hermite (simétrica y definida positiva)."""
    h = 1 / (n + 1)
    D = np.array([[72, 0], [0, 8 * h**2]]) / (30 * h)
    L = np.array([[-36, -3 * h], [3 * h, -h**2]]) / (30 * h)
    return bloques(D, L, L.T, n)


def matriz_punto3(n):
    return matriz_tercer_orden(n) + matriz_rigidez_K1(n)


if __name__ == "__main__":
    print("Punto 3:  u''' - u'' = f")
    hs, E3 = estudio(matriz_punto3, f, uex)
    c = figura_solucion(matriz_punto3, f, uex, "p3_solucion")

    # Solución aproximada con n = 5: coeficientes u_h(x_j) y u_h'(x_j)
    n, h = 5, 1 / 6
    xj = np.arange(1, n + 1) * h
    duex = lambda x: x * (1 - x) * (2 - 3 * x - x**2) * np.exp(x)  # derivada de la solución exacta
    print(f"\n{'x_j':>6} {'u_h(x_j)':>11} {'u_ex(x_j)':>11} {'du_h(x_j)':>11} {'du_ex(x_j)':>11}")
    for j in range(n):
        print(f"{xj[j]:>6.3f} {c[2*j]:>11.6f} {uex(xj[j]):>11.6f} {c[2*j+1]:>11.6f} {duex(xj[j]):>11.6f}")

    print("\nPunto 2 (para comparar):  u''' = f")
    estudio(matriz_tercer_orden, f2, uex)

    # Parte simétrica de cada matriz: (A + A^T)/2. Su menor autovalor dice si x^T A x > 0.
    for nombre, matriz in [("Punto 2", matriz_tercer_orden), ("Punto 3", matriz_punto3)]:
        A = matriz(10)
        lam = np.linalg.eigvalsh((A + A.T) / 2)
        print(f"{nombre}: autovalores de la parte simétrica en [{lam.min():.2e}, {lam.max():.2e}]")
