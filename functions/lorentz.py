import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


def lorenz_runge_kutta_4(dt, step, y0s, params):
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

def lorentz_equation(xyz, p, r, b):
    return [
        -p * xyz[0] + p * xyz[1],
        -xyz[0] * xyz[2] + r * xyz[0] - xyz[1],
        xyz[0] * xyz[1] - b * xyz[2]
    ]

def plot_lorentz(res):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Lorenz attractor (Runge-Kutta method)")

    n = len(res[0])
    c = np.linspace(0,1,n)
    pas = int(n/10)

    x, y, z = res
    for i in range(0, n-pas, pas):
        ax.plot(x[i:i+pas+1], y[i:i+pas+1], z[i:i+pas+1], color=(1, c[i],0), lw=1)
    plt.show()
