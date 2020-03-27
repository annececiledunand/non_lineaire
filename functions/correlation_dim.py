import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as dist
from statsmodels.tsa.tsatools import lagmat
from tqdm import tqdm

from functions import plongement

# Pour les fonctions find_tau et find_d_embedding, j'ai pris comme inspiration
# le code de : https://github.com/jcvasquezc/Corr_Dim/blob/master/Corr_Dim.py
# qui est lui meme inspire des fonctions matlab associees


def entropy(data: list, d: int) -> float:
    data = np.asarray(data)
    n = len(data)
    tolerance = 0.2 * np.std(data)

    time_vect = plongement.plongement(np.array(data), d=d+1, tau=1)
    time_vect = np.array(time_vect)

    plot_data = []
    counts = []
    for m in [d, d + 1]:
        counts.append(0)
        time_vect_M = time_vect[:n-m+1,:m]

        for i in range(len(time_vect_M) - 1):
            dsts = np.max(np.abs(time_vect_M[i + 1:] - time_vect_M[i]), axis=1)
            counts[-1] += np.sum(dsts <= tolerance)

    if counts[0] > 0 and counts[1] > 0:
        entropy = -np.log(1.0 * counts[1] / counts[0])
    return entropy


def correlation_dim(datas: List[np.ndarray], tau: int, d: int) -> float:
    x = phase_space(datas, d, tau)

    print('Finding correlation dimension ...')
    ED2 = dist(x.T)
    posD = np.triu_indices_from(ED2, k=1)
    ED = ED2[posD]

    max_eps = np.max(ED)
    min_eps = np.min(ED[np.where(ED>0)])
    max_eps = np.exp(math.floor(np.log(max_eps)))

    n_div = int(math.floor(np.log(max_eps/min_eps)))
    n_eps = n_div+1
    eps_vec = range(n_eps)
    unos = np.ones([len(eps_vec)])*-1

    eps_vec1 = max_eps*np.exp(unos*eps_vec-unos)
    Npairs = ((len(x[1,:]))*((len(x[1,:])-1)))
    C_eps = np.zeros(n_eps)

    for i in eps_vec:
        eps = eps_vec1[i]
        N = np.where(((ED<eps) & (ED>0)))
        S = len(N[0])
        C_eps[i] = float(S)/Npairs

    omit_pts=1
    k1 = omit_pts
    k2 = n_eps-omit_pts
    xd = np.log(eps_vec1)
    yd = np.log(C_eps)
    xp = xd[k1:k2]
    yp = yd[k1:k2]
    p = np.polyfit(xp, yp, 1)
    return p[0]


def find_d_embedding(data: list, maxm: int) -> int:
    RT = 15.0
    AT = 2

    sigmay = np.std(data, ddof=1)
    nyr = len(data)
    m = maxm

    EM = lagmat(data, maxlag=m-1)
    EEM = np.asarray([EM[j,:] for j in range(m-1, EM.shape[0])])
    embedm = maxm

    for k in range(AT,EEM.shape[1]+1):
        fnn1 = []
        fnn2 = []
        Ma = EEM[:,range(k)]
        D = dist(Ma)

        for i in range(1,EEM.shape[0]-m-k):
            d = D[i,:]
            pdnz = np.where(d>0)
            dnz = d[pdnz]
            Rm = np.min(dnz)

            l = np.where(d==Rm)
            l = l[0]
            l = l[len(l)-1]

            if l+m+k-1 < nyr:
                fnn1.append(np.abs(data[i+m+k-1]-data[l+m+k-1])/Rm)
                fnn2.append(np.abs(data[i+m+k-1]-data[l+m+k-1])/sigmay)

        Ind1 = np.where(np.asarray(fnn1)>RT)
        Ind2 = np.where(np.asarray(fnn2)>AT)

        if len(Ind1[0])/float(len(fnn1))<0.1 and len(Ind2[0])/float(len(fnn2))<0.1:
            embedm = k
            break

    return embedm


def find_tau(data: list) -> int:
    corr = np.correlate(data, data, mode="full")
    corr = corr[int(len(corr)/2):len(corr)]
    tau = 0
    j = 0
    while (corr[j]>0):
    	j += 1
    tau = j
    return tau


def phase_space(data: list, d : int, tau: int) -> np.ndarray:
    n = len(data)
    x = np.zeros([d, (n-(d-1)*tau)])

    for j in range(d):
        l1 = (tau*(j))
        l2 = (tau*(j) + len(x[1,:]))
        x[j,:] = data[l1:l2]

    return x
