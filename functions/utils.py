import matplotlib.pyplot as plt
from typing import Callable, List

import numpy as np
from tqdm import tqdm


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


def get_section_poincare(x: list, y: list, z: list, epsilon: float) -> List[list]:
    y_poincare, z_poincare = [], []
    for i, xi in enumerate(x):
        if abs(xi) <= epsilon :
            y_poincare.append(y[i])
            z_poincare.append(z[i])

    return y_poincare, z_poincare


def plot_section_poincare(x: list, y: list, z: list, epsilon: float) -> None:
    y_poincare, z_poincare = get_section_poincare(x, y, z, epsilon)

    fig = plt.figure()
    plt.scatter(y_poincare, z_poincare, s=0.2, color='red')
    plt.title(f'Poincaré section avec epsilon = {epsilon}, nb de points : {len(y_poincare)}')
    plt.show()
