"""Punto 1b: u_t - u_xx + u = f en (0,T) x (0,1), u(t,0) = u(t,1) = 0, u(0,x) = u0(x).

Espacio: elementos P2 del punto 1a.  Tiempo: Euler implícito (diferencias finitas hacia atrás).
Solución exacta propuesta: u(t,x) = e^{-t} x(1-x) e^x  ->  f(t,x) = e^{-t} x(x+3) e^x.
"""
import numpy as np
import matplotlib.pyplot as plt
from punto1a import matrices_P2, carga_P2, errores_P2


def uex(t, x):
    return np.exp(-t) * x * (1 - x) * np.exp(x)


def f(t, x):
    return np.exp(-t) * x * (x + 3) * np.exp(x)


def calor_P2(n, dt, T=1.0):
    """Devuelve los tiempos, las soluciones U^k y los errores (L2, Linf) en cada paso."""
    h = 1 / (n + 1)
    K, M = matrices_P2(n)
    A = M + dt * (K + M)                                     # (M + dt(K + M)) U^{k+1} = M U^k + dt F^{k+1}
    xv = np.arange(1, n + 1) * h                             # vértices interiores
    xm = (np.arange(n + 1) + 0.5) * h                        # puntos medios
    U = uex(0, np.concatenate([xv, xm]))                     # U^0: interpolante de u0
    N = int(round(T / dt))
    ts, sols, errs = [0.0], [U], [errores_P2(n, U, lambda x: uex(0, x))]
    for k in range(1, N + 1):
        t = k * dt
        U = np.linalg.solve(A, M @ U + dt * carga_P2(n, lambda x: f(t, x)))
        ts.append(t); sols.append(U); errs.append(errores_P2(n, U, lambda x: uex(t, x)))
    return np.array(ts), sols, np.array(errs)


if __name__ == "__main__":
    n, dt = 37, 0.01
    ts, sols, errs = calor_P2(n, dt)

    print(f"n = {n}, h = {1/(n+1):.5f}, dt = {dt}")
    print(f"{'paso k':>7} {'t_k':>6} {'error L2':>12} {'error Linf':>12}")
    for k in range(0, len(ts), 10):
        print(f"{k:>7} {ts[k]:>6.2f} {errs[k, 0]:>12.4e} {errs[k, 1]:>12.4e}")

    # Error en t = T al refinar dt (n fijo): muestra el orden en tiempo
    print(f"\n{'dt':>8} {'error L2 en T=1':>16} {'orden':>7}")
    dts, eT = [0.04, 0.02, 0.01, 0.005], []
    for d in dts:
        eT.append(calor_P2(n, d)[2][-1, 0])
    for k, d in enumerate(dts):
        p = f"{np.log(eT[k-1] / eT[k]) / np.log(2):.3f}" if k > 0 else ""
        print(f"{d:>8} {eT[k]:>16.4e} {p:>7}")

    # Figura 1: solución aproximada y exacta en tres instantes
    h = 1 / (n + 1)
    xv = np.arange(0, n + 2, 2) * h                           # un vértice de cada dos
    xf = np.linspace(0, 1, 300)
    plt.figure(figsize=(6, 3.6))
    for t, col in [(0.0, "#2a78d6"), (0.5, "#eb6834"), (1.0, "#1baf7a")]:
        k = int(round(t / dt))
        plt.plot(xf, uex(t, xf), color=col, lw=2, label=rf"$u_{{ex}}$, $t={t}$")
        uv = np.concatenate([[0], sols[k][:n], [0]])            # u_h en los vértices
        plt.plot(xv, uv[::2], "o", ms=4, color=col, label=rf"$u_h$, $t={t}$")
    plt.xlabel("$x$"); plt.legend(ncol=3, fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("../figuras/p1b_solucion.png", dpi=300)

    # Figura 2: error L2 a lo largo del tiempo para varios dt
    plt.figure(figsize=(6, 3.6))
    for d, col in [(0.02, "#2a78d6"), (0.01, "#eb6834"), (0.005, "#1baf7a")]:
        t_d, _, e_d = calor_P2(n, d)
        plt.plot(t_d, e_d[:, 0], color=col, lw=2, label=rf"$\Delta t={d}$")
    plt.xlabel("$t$"); plt.ylabel(r"$\|u_{ex}(t)-u_h(t)\|_{L^2}$")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("../figuras/p1b_error_tiempo.png", dpi=300)
