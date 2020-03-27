import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List

import numpy as np
from tqdm import tqdm


def plongement(x: list, d: int, tau: int) -> list:
    n = len(x)
    m = (d-1)*tau

    y = []
    for i in range(m, n):
        xn = []
        for j in range(d):
            xn.append(x[i - (d-j-1)*tau])
        xn.reverse()
        y.append(xn)

    return y


def animation_plongement_2d(x: list, tau_pas: int, tau_max: int) -> None:
    fig, ax = plt.subplots()
    Xs = []
    for tau in range(0, tau_max, tau_pas):
        Xs.append(plongement(x, d=2, tau=tau))

    line, = ax.plot([], [], lw=1)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)

    ax.set_title('Plongement d=2 evolution avec tau')
    ax.set_xlabel('x_n')
    ax.set_ylabel('x_(n-tau)')

    def init():
        line.set_data([],[])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        thisx = [el[0] for el in Xs[i]]
        thisy = [el[1] for el in Xs[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(f'tau = {i*tau_pas}')
        return line, time_text

    ani = animation.FuncAnimation(
        fig,
        animate,
        range(int(tau_max/tau_pas)),
        blit=True,
        init_func=init,
        interval=500,
        repeat=False
    )
    plt.show()


def plot_plongement_2d(x: list, time_delay: int) -> None:
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Plongement d=2, tau={time_delay}")

    n = len(x)
    c = np.linspace(0,1,n)
    pas = int(n/10)

    for i in range(0, n-pas, pas):
        ax.plot([el[0] for el in x[i:i+pas+1]], [el[1] for el in  x[i:i+pas+1]], color=(1, c[i],0), lw=1)

    plt.show()
