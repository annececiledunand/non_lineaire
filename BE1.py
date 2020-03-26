import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, List
from tqdm import tqdm
from scipy.integrate import odeint, RK45


# Analyse de données temporelles

def filtre_mobile(x: list, N: int = 1024) -> np.ndarray:
    y = []

    for n in range(len(x) -7):
        sum7 = sum(x[n:n+14])
        y.append(sum7)

    return y[7:-7]

def DSP(data: np.ndarray, time_step: float) -> List[np.ndarray]:
    ps = np.abs(np.fft.fft(data, n=data.shape[0]))**2

    freqs = np.fft.fftfreq(data.size, time_step)
    idx = np.argsort(freqs)

    return freqs, ps, idx


def processus_ulam(N: int, y0: float, rec_relation: Callable) -> list:
    yn = [y0]
    for i in range(N - 1):
        yn.append(rec_relation(yn[i]))

    return np.asarray(yn)


def fct_observation_sn(signal_x: list) -> list:
    return np.asarray([(1./np.pi) * np.arccos(xn) for xn in signal_x])


# Plot lorentz attractor:

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
    ax.plot(res[0], res[1], res[2], color="red", lw=1)
    plt.show()


def Exercice1():
    print('# Question 1 & 2 \n')

    N = 1024
    x = np.random.normal(loc=0, scale=1, size=(N + 14))
    y = np.asarray(filtre_mobile(x))

    for data, label in [(x, 'xn'), (y, 'yn')]:
        freqs, ps, idx = DSP(data, 0.01)
        max_ps = max(ps)
        plt.plot(freqs[idx], [p/max_ps for p in ps[idx]], label=label)

    plt.legend()
    plt.show()

    print('# Question 3 \n')
    N = 4096

    xn = np.random.normal(loc=0, scale=1, size=N)
    yn = processus_ulam(N, y0=0.1, rec_relation=lambda x: 1 - 2*x*x)
    sn = fct_observation_sn(yn)

    for data, label in [(xn, 'xn'), (yn, 'yn'), (sn, 'sn')]:
        print(f'Valeurs pour {label} :')
        print(f'\tMoyenne  : {np.mean(data)}')
        print(f'\tVariance : {np.var(data)}')

        freqs, ps, idx = DSP(data, 0.01)
        max_ps = max(ps)
        plt.plot(freqs[idx], [p/max_ps for p in ps[idx]], label=label)

    # On a égalité parfaite des DSP de xn et sn
    plt.legend()
    plt.show()


def Exercice3():
    print('# Question 1\n')

    dt = 1e-3
    step = 100000
    res = lorenz_runge_kutta_4(dt, step, [1., 1., 1.], [10, 28, 8/3.0])
    plot_lorentz(res)

    print('# Question 2\n')



if __name__ == '__main__':
    Exercice3()
