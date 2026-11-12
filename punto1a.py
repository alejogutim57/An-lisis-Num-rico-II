"""Punto 1a: elementos finitos P2 (Lagrange) para -u'' = f en (0,1), u(0) = u(1) = 0.

Solución exacta propuesta: u(x) = x(1-x)e^x  ->  f(x) = x(x+3)e^x.
Malla uniforme con nodos x_0 = 0, ..., x_{n+1} = 1 (n nodos interiores), h = 1/(n+1).
Incógnitas: n valores en vértices interiores y n+1 valores en puntos medios.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Carpeta de salida independiente del directorio de ejecución.
FIGURAS = Path(__file__).resolve().parent / "figuras"
FIGURAS.mkdir(parents=True, exist_ok=True)


def uex(x):
    return x * (1 - x) * np.exp(x)


def f(x):
    return x * (x + 3) * np.exp(x)


# Cuadratura de Gauss-Legendre de 4 puntos llevada al intervalo [0, 1]
g, wg = np.polynomial.legendre.leggauss(4)
s, w = (g + 1) / 2, wg / 2


def base_P2(s):
    """Funciones base sobre un elemento, en la coordenada local s = (x - x_j)/h:
    vértice izquierdo phi_j, punto medio psi_{j+1/2} y vértice derecho phi_{j+1}."""
    return np.array([(1 - s) * (1 - 2 * s), 1 - 4 * (s - 0.5) ** 2, s * (2 * s - 1)])


def matrices_P2(n):
    """Matrices de rigidez K y de masa M, por bloques (vértices | puntos medios)."""
    h = 1 / (n + 1)
    I, J = np.eye(n), np.eye(n + 1)
    T = np.eye(n, k=1) + np.eye(n, k=-1)               # vecinos entre vértices
    B = np.eye(n, n + 1) + np.eye(n, n + 1, k=1)       # vértice x_i con x_{i-1/2} y x_{i+1/2}
    K = np.block([[(14 * I + T) / (3 * h), -8 * B / (3 * h)],
                  [-8 * B.T / (3 * h), 16 * J / (3 * h)]])
    M = h / 30 * np.block([[8 * I - T, 2 * B],
                           [2 * B.T, 16 * J]])
    return K, M


def carga_P2(n, f):
    """Vector de carga: F_i = int f phi_i (vértices) y G_j = int f psi_{j+1/2} (puntos medios)."""
    h = 1 / (n + 1)
    Fv, Fm = np.zeros(n + 2), np.zeros(n + 1)
    N = base_P2(s)
    for e in range(n + 1):                              # elemento [x_e, x_{e+1}]
        fw = f((e + s) * h) * w * h
        Fv[e] += fw @ N[0]
        Fm[e] += fw @ N[1]
        Fv[e + 1] += fw @ N[2]
    return np.concatenate([Fv[1:-1], Fm])               # se quitan x_0 y x_{n+1}


def evaluar_P2(n, U, s_loc):
    """Valores de u_h en los puntos s_loc de cada elemento."""
    vert = np.concatenate([[0], U[:n], [0]])            # u_h(x_j), con u_h(0) = u_h(1) = 0
    med = U[n:]                                         # u_h(x_{j+1/2})
    N = base_P2(s_loc)
    return [vert[e] * N[0] + med[e] * N[1] + vert[e + 1] * N[2] for e in range(n + 1)]


def errores_P2(n, U, u):
    """Error en L2 (cuadratura de Gauss) y en L-infinito (muestreo fino)."""
    h = 1 / (n + 1)
    ss = np.linspace(0, 1, 21)
    uh_g, uh_f = evaluar_P2(n, U, s), evaluar_P2(n, U, ss)
    eL2 = np.sqrt(sum(np.sum(w * h * (u((e + s) * h) - uh_g[e]) ** 2) for e in range(n + 1)))
    eInf = max(np.max(np.abs(u((e + ss) * h) - uh_f[e])) for e in range(n + 1))
    return eL2, eInf


if __name__ == "__main__":
    ns = [5, 10, 20, 50, 100]
    hs, eL2s, eInfs = [], [], []
    for n in ns:
        K, _ = matrices_P2(n)
        U = np.linalg.solve(K, carga_P2(n, f))
        eL2, eInf = errores_P2(n, U, uex)
        hs.append(1 / (n + 1)); eL2s.append(eL2); eInfs.append(eInf)

    print(f"{'n':>5} {'h':>10} {'error L2':>12} {'orden':>7} {'error Linf':>12} {'orden':>7}")
    for k, n in enumerate(ns):
        p2 = p8 = ""
        if k > 0:
            r = np.log(hs[k - 1] / hs[k])
            p2 = f"{np.log(eL2s[k - 1] / eL2s[k]) / r:.3f}"
            p8 = f"{np.log(eInfs[k - 1] / eInfs[k]) / r:.3f}"
        print(f"{n:>5} {hs[k]:>10.5f} {eL2s[k]:>12.4e} {p2:>7} {eInfs[k]:>12.4e} {p8:>7}")

    # Figura 1: solución aproximada con n = 5 frente a la exacta
    n = 5
    K, _ = matrices_P2(n)
    U = np.linalg.solve(K, carga_P2(n, f))
    h = 1 / (n + 1)
    ss = np.linspace(0, 1, 30)
    xx = np.concatenate([(e + ss) * h for e in range(n + 1)])
    uh = np.concatenate(evaluar_P2(n, U, ss))
    xv = np.linspace(0, 1, n + 2)
    xm = (np.arange(n + 1) + 0.5) * h
    plt.figure(figsize=(6, 3.6))
    plt.plot(xx, uex(xx), color="#2a78d6", lw=2, label=r"$u_{ex}(x)$")
    plt.plot(xx, uh, "--", color="#eb6834", lw=2, label=r"$u_h(x)$, $n=5$")
    plt.plot(xv, np.concatenate([[0], U[:n], [0]]), "o", color="#eb6834", label="vértices")
    plt.plot(xm, U[n:], "s", mfc="white", color="#eb6834", label="puntos medios")
    plt.xlabel("$x$"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIGURAS / "p1a_solucion.png", dpi=300)

    # Figura 2: error frente a h en escala log-log
    plt.figure(figsize=(6, 3.6))
    plt.loglog(hs, eL2s, "o-", color="#2a78d6", label=r"$\|u_{ex}-u_h\|_{L^2}$")
    plt.loglog(hs, eInfs, "s-", color="#eb6834", label=r"$\|u_{ex}-u_h\|_{L^\infty}$")
    hh = np.array(hs)
    plt.loglog(hh, eL2s[0] * (hh / hh[0]) ** 3, ":", color="gray", label=r"referencia $h^3$")
    plt.xlabel("$h$"); plt.ylabel("error"); plt.legend(); plt.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig(FIGURAS / "p1a_error.png", dpi=300)
