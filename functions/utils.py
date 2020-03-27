import numpy as np
from typing import Callable, List
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
