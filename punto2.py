"""Punto 2: elementos finitos P3 (Hermite) para u''' = f en (0,1),
u(0) = u(1) = 0, u'(0) = u'(1) = 0.

Formulación variacional: hallar u en H^2_0 con  a(u,v) = -int u'' v' = int f v.
Solución exacta propuesta: u(x) = x^2 (1-x)^2 e^x  ->  f(x) = (x^4 + 10x^3 + 19x^2 - 6x - 6) e^x.
Incógnitas por nodo interior x_j: (u_j, u'_j). Se usa psi_j escalada por h para que el
coeficiente que acompaña a psi_j sea exactamente u_h'(x_j).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Carpeta de salida independiente del directorio de ejecución.
FIGURAS = Path(__file__).resolve().parent / "figuras"
FIGURAS.mkdir(parents=True, exist_ok=True)


def uex(x):
    return x**2 * (1 - x)**2 * np.exp(x)


def f(x):
    return (x**4 + 10 * x**3 + 19 * x**2 - 6 * x - 6) * np.exp(x)


g, wg = np.polynomial.legendre.leggauss(5)
s, w = (g + 1) / 2, wg / 2


def base_hermite(s, h):
    """Base local en [x_j, x_{j+1}]: valor y pendiente a la izquierda, valor y pendiente a la derecha."""
    return np.array([(1 - s)**2 * (1 + 2 * s), h * s * (1 - s)**2,
                     s**2 * (3 - 2 * s), h * s**2 * (s - 1)])


def bloques(D, L, U, n):
    """Matriz 2n x 2n tridiagonal por bloques 2x2: D en la diagonal, L abajo, U arriba."""
    return np.kron(np.eye(n), D) + np.kron(np.eye(n, k=-1), L) + np.kron(np.eye(n, k=1), U)


def matriz_tercer_orden(n):
    """Matriz de a(u,v) = -int u'' v'  (es antisimétrica)."""
    h = 1 / (n + 1)
    D = np.array([[0, -2 / h], [2 / h, 0]])
    L = np.array([[0, 1 / h], [-1 / h, -0.5]])
    U = np.array([[0, 1 / h], [-1 / h, 0.5]])
    return bloques(D, L, U, n)


def carga_hermite(n, f):
    """F_{2j} = int f phi_j  y  F_{2j+1} = int f psi_j, sobre los nodos interiores."""
    h = 1 / (n + 1)
    F = np.zeros(2 * (n + 2))
    N = base_hermite(s, h)
    for e in range(n + 1):
        F[2 * e: 2 * e + 4] += N @ (f((e + s) * h) * w * h)
    return F[2:-2]


def evaluar_hermite(n, c, s_loc):
    h = 1 / (n + 1)
    c = np.concatenate([[0, 0], c, [0, 0]])               # u y u' nulos en x = 0 y x = 1
    N = base_hermite(s_loc, h)
    return [c[2 * e: 2 * e + 4] @ N for e in range(n + 1)]


def errores_hermite(n, c, u):
    h = 1 / (n + 1)
    ss = np.linspace(0, 1, 21)
    uh_g, uh_f = evaluar_hermite(n, c, s), evaluar_hermite(n, c, ss)
    eL2 = np.sqrt(sum(np.sum(w * h * (u((e + s) * h) - uh_g[e])**2) for e in range(n + 1)))
    eInf = max(np.max(np.abs(u((e + ss) * h) - uh_f[e])) for e in range(n + 1))
    return eL2, eInf


def estudio(matriz, f, u, ns=(5, 10, 20, 50, 100)):
    """Resuelve para cada n e imprime la tabla de errores, órdenes y número de condición."""
    hs, E = [], []
    print(f"{'n':>5} {'error L2':>12} {'orden':>7} {'error Linf':>12} {'orden':>7} {'cond(A)':>10}")
    for k, n in enumerate(ns):
        A = matriz(n)
        c = np.linalg.solve(A, carga_hermite(n, f))
        hs.append(1 / (n + 1)); E.append(errores_hermite(n, c, u))
        p2 = p8 = ""
        if k > 0:
            r = np.log(hs[k - 1] / hs[k])
            p2 = f"{np.log(E[k-1][0] / E[k][0]) / r:.3f}"
            p8 = f"{np.log(E[k-1][1] / E[k][1]) / r:.3f}"
        print(f"{n:>5} {E[k][0]:>12.4e} {p2:>7} {E[k][1]:>12.4e} {p8:>7} {np.linalg.cond(A):>10.2e}")
    return np.array(hs), np.array(E)


def figura_solucion(matriz, f, u, nombre, n=5):
    h = 1 / (n + 1)
    c = np.linalg.solve(matriz(n), carga_hermite(n, f))
    ss = np.linspace(0, 1, 30)
    xx = np.concatenate([(e + ss) * h for e in range(n + 1)])
    plt.figure(figsize=(6, 3.6))
    plt.plot(xx, u(xx), color="#2a78d6", lw=2, label=r"$u_{ex}(x)$")
    plt.plot(xx, np.concatenate(evaluar_hermite(n, c, ss)), "--", color="#eb6834", lw=2,
             label=rf"$u_h(x)$, $n={n}$")
    plt.plot(np.linspace(0, 1, n + 2), np.concatenate([[0], c[0::2], [0]]), "o", color="#eb6834")
    plt.xlabel("$x$"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIGURAS / f"{nombre}.png", dpi=300)
    return c


if __name__ == "__main__":
    hs, E = estudio(matriz_tercer_orden, f, uex)
    figura_solucion(matriz_tercer_orden, f, uex, "p2_solucion")

    A = matriz_tercer_orden(5)
    print("\n¿A es antisimétrica?", np.allclose(A, -A.T))

    plt.figure(figsize=(6, 3.6))
    plt.loglog(hs, E[:, 0], "o-", color="#2a78d6", label=r"$\|u_{ex}-u_h\|_{L^2}$")
    plt.loglog(hs, E[:, 1], "s-", color="#eb6834", label=r"$\|u_{ex}-u_h\|_{L^\infty}$")
    plt.loglog(hs, E[0, 0] * (hs / hs[0])**2, ":", color="gray", label=r"referencia $h^2$")
    plt.xlabel("$h$"); plt.ylabel("error"); plt.legend(); plt.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig(FIGURAS / "p2_error.png", dpi=300)
