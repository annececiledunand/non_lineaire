from math import sqrt, log
from typing import List

import numpy as np
from tqdm import tqdm


def phase_space_reconstruction(x: np.ndarray, m: int, time_delay: int, nb_recon_vect: int = -1) -> np.ndarray:
    N = len(x)
    if nb_recon_vect == -1:
        nb_recon_vect = N - (m-1)*time_delay

    y = np.zeros((nb_recon_vect, m))
    for i in range(m):
        x_flat = x.flatten(order='F')
        y[:, i] = x_flat[i*time_delay:nb_recon_vect+i*time_delay]
    return y


def lyapunov_rosenstein(x: np.ndarray, m: int, time_delay: int, mean_period: float, max_iter: int) -> list:
    N = len(x)
    M = N - (m-1)*time_delay
    near_value, near_index, d = [], [], []

    y = phase_space_reconstruction(x, m, time_delay)

    for i in range(M):
        x0 = np.ones((M, 1)) * y[i,:]
        distance = list(np.sqrt(np.sum((y - x0)**2, axis=1)))

        for j in range(M):
            if abs(j-i) <= mean_period:
                distance[j] = 1e10

        near_value.append(min(distance))
        near_index.append(distance.index(near_value[i]))

    for k in range(max_iter):
        max_ind = M - k - 1
        evolve = 0
        pnt = 0

        for j in range(M):
            if (j+1 <= max_ind) and (near_index[j] <= max_ind):
                dist_k = np.sqrt(np.sum((y[j+k, :] - y[near_index[j] + k, :])**2))
                if dist_k != 0.:
                    evolve = evolve + log(dist_k)
                    pnt += 1

        if pnt > 0:
            d.append(evolve/pnt)
        else:
            d.append(0)

    return d
