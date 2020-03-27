import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List

import numpy as np
from tqdm import tqdm


###### Lorentz ######


def lorenz_runge_kutta_4(dt: float, step: int, y0s: List[float], params: list) -> List[list]:
    X_0, Y_0, Z_0 = y0s
    res = [[], [], []]

    xyz = [X_0, Y_0, Z_0]
    for _ in range(step):
        k_0 = lorentz_equation(xyz, *params)
        k_1 = lorentz_equation([x + k * dt/2 for x, k in zip(xyz, k_0)], *params)
        k_2 = lorentz_equation([x + k * dt/2 for x, k in zip(xyz, k_1)], *params)
        k_3 = lorentz_equation([x + k * dt for x, k in zip(xyz, k_2)], *params)

        for i in range(3):
            xyz[i] += (k_0[i] + 2*k_1[i] + 2*k_2[i] + k_3[i]) * dt/6.0
            res[i].append(xyz[i])

    return res


def lorentz_equation(xyz: List[list], p: float, r: float, b: float) -> List[list]:
    return [
        -p * xyz[0] + p * xyz[1],
        -xyz[0] * xyz[2] + r * xyz[0] - xyz[1],
        xyz[0] * xyz[1] - b * xyz[2]
    ]


def plot_lorentz(res: List[list], params: List[float]) -> None:
    p, r, b = params
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Lorenz attractor rho={r}, sigma={p}, beta={round(b, 2)}")

    n = len(res[0])
    c = np.linspace(0, 1, n)
    pas = int(n/10)

    x, y, z = res
    for i in range(0, n-pas, pas):
        ax.plot(x[i:i+pas+1], y[i:i+pas+1], z[i:i+pas+1], color=(1, c[i],0), lw=1)
    plt.show()


###### Henon ######


def henon_rec(x: float, y: float, a: float, b: float) -> list:
    return 1 - a*x*x + y, b*x


def henon(x0: float, y0: float, a: float, b: float, N: int) -> List[list]:
    x, y = [x0], [y0]
    for i in range(N-1):
        xn, yn = henon_rec(x[i], y[i], a, b)
        x.append(xn)
        y.append(yn)

    return x, y


def plot_henon(x: list, y: list, params: List[float]) -> None:
    a, b, N = params
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Henon attractor a={a}, b={b}, N={N}")

    ax.scatter(x, y, color='red', s=0.2)
    plt.show()


###### Baier-Klein ######


def relation_baier_klein(x: list, y: list, z: list) -> List[list]:
    return 1.76 - y*y - 0.1*z, x, y


def baier_klein(x0: float, y0: float, z0: float, N: int) -> List[list]:
    x, y, z = [x0], [y0], [z0]

    for i in range(N - 1):
        xn, yn, zn = relation_baier_klein(x[i], y[i], z[i])
        x.append(xn)
        y.append(yn)
        z.append(zn)

    return x, y, z


def plot_baier_klein(res: list) -> None:
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Baier Klein System")
    ax.scatter(res[0], res[1], res[2], color="red", s=0.4)
    plt.show()


###### Ikeda ######


def ikeda_relation(z: float) -> complex:
    complex_z = 0.4j - 6.j/(1 + abs(z)*abs(z))
    return 1 + 0.9*z*np.exp(complex_z)


def ikeda(z0: float, N: int) -> complex:
    z = [z0]

    for i in range(N-1):
        z.append(ikeda_relation(z[i]))

    return z
