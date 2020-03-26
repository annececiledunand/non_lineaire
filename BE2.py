import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, List
from tqdm import tqdm
from scipy.integrate import odeint, RK45

import math

from math import log
from statsmodels.tsa.tsatools import lagmat
from sklearn.metrics.pairwise import euclidean_distances as dist

import BE1

#
# class LorenzMap:
#     def __init__(_, sigma=10, rho=28, beta=8 / 3, h0=0.01):
#         _.sigma, _.rho, _.beta = sigma, rho, beta
#         _.h0 = h0
#
#     @staticmethod
#     def pack_variables(xyz, w):
#         return np.concatenate((xyz, np.reshape(w, 9)), axis=0)
#
#     @staticmethod
#     def unpack_variables(xyzw):
#         return xyzw[0:3], np.reshape(xyzw[3::], (3, 3))
#
#     def variational_equation(_, xyzw, t=None):
#         xyz, w = _.unpack_variables(xyzw)
#         x, y, z = xyz
#
#         dot_xyz = np.array([_.sigma * (-x + y),
#                          x * (_.rho - z) - y,
#                          x * y - _.beta * z])
#
#         dot_w = np.array([
#                         [-_.sigma, _.sigma, 0],
#                         [_.rho - z, -1, -x],
#                         [y, x, -_.beta]
#                        ]) @ w
#
#         return _.pack_variables(dot_xyz, dot_w)
#
#     def __call__(_, xyz, w):
#         xyzw = _.pack_variables(xyz, w)
#         next_xyzw = odeint(_.variational_equation, xyzw, np.array([0, 1]), h0=_.h0)
#         return _.unpack_variables(next_xyzw[1])
#
# class BaierKleinMap:
#     def __init__(_, h0=0.01):
#         _.h0 = h0
#
#     @staticmethod
#     def pack_variables(xyz, w):
#         return np.concatenate((xyz, np.reshape(w, 9)), axis=0)
#
#     @staticmethod
#     def unpack_variables(xyzw):
#         return xyzw[0:3], np.reshape(xyzw[3::], (3, 3))
#
#     def variational_equation(_, xyzw, t=None):
#         xyz, w = _.unpack_variables(xyzw)
#         x, y, z = xyz
#
#         dot_xyz = np.array([1.76 - y*y - 0.1*z, x, y])
#
#         dot_w = np.array([
#                         [0, -2*y, -0.1],
#                         [1, 0, 0],
#                         [0, 1, 0]
#                        ]) @ w
#
#         return _.pack_variables(dot_xyz, dot_w)
#
#     def __call__(_, xyz, w):
#         xyzw = _.pack_variables(xyz, w)
#         next_xyzw = odeint(_.variational_equation, xyzw, np.array([0, 1]), h0=_.h0)
#         return _.unpack_variables(next_xyzw[1])
#
# def lyapunov_exponent(f_df, initial_condition, tol=0.1, max_it=1000, min_it_percentage=0.1):
#     x = initial_condition
#     n = len(initial_condition)
#     w = np.eye(n)
#     h = np.zeros(n)
#     trans_it = int(max_it * min_it_percentage)
#     l = -1
#
#     for i in range(0, max_it):
#         x_next, w_next = f_df(x, w)
#         w_next = orthogonalize_columns(w_next)
#
#         h_next = h + log_of_the_norm_of_the_columns(w_next)
#         l_next = h_next / (i + 1)
#
#         if i > trans_it and np.linalg.norm(l_next - l) < tol:
#             return np.sort(l_next)
#
#         h = h_next
#         x = x_next
#         w = normalize_columns(w_next)
#         l = l_next
#
#     raise Exception('Lyapunov Exponents computation did no convergence')
#
# def orthogonalize_columns(a):
#     q, r = np.linalg.qr(a)
#     return q @ np.diag(r.diagonal())
#
# def normalize_columns(a):
#     return np.apply_along_axis(lambda v: v / np.linalg.norm(v), 0, a)
#
# def log_of_the_norm_of_the_columns(a):
#     return np.apply_along_axis(lambda v: np.log(np.linalg.norm(v)), 0, a)



# def Dim_Corr(x):
#     ED2=dist(x.T)
#     posD=np.triu_indices_from(ED2, k=1)
#     ED=ED2[posD]
#
#     max_eps=np.max(ED)
#     min_eps=np.min(ED[np.where(ED>0)])
#     max_eps=np.exp(math.floor(np.log(max_eps)))
#
#     n_div=int(math.floor(np.log(max_eps/min_eps)))
#     n_eps=n_div+1
#
#     eps_vec=range(n_eps)
#     unos=np.ones([len(eps_vec)])*-1
#
#     eps_vec1=max_eps*np.exp(unos*eps_vec-unos)
#     Npairs=((len(x[1,:]))*((len(x[1,:])-1)))
#     C_eps=np.zeros(n_eps)
#
#     for i in eps_vec:
#         eps=eps_vec1[i]
#         N=np.where(((ED<eps) & (ED>0)))
#         S=len(N[0])
#         C_eps[i]=float(S)/Npairs
#
#     omit_pts=1
#
#     k1=omit_pts
#     k2=n_eps-omit_pts
#
#     xd=np.log(eps_vec1)
#     yd=np.log(C_eps)
#     xp=xd[k1:k2]
#     yp=yd[k1:k2]
#
#     p = np.polyfit(xp, yp, 1)
#     return p[0]
#
# def PhaseSpace(data, m, Tao, graph=False):
#     ld=len(data)
#     x = np.zeros([m, (ld-(m-1)*Tao)])
#     for j in range(m):
#         l1=(Tao*(j))
#         l2=(Tao*(j)+len(x[1,:]))
#         x[j,:]=data[l1:l2]
#     if graph:
#         fig = plt.figure()
#         if m>2:
#             ax = fig.add_subplot(111, projection='3d')
#             ax.plot(x[0,:], x[1,:], x[2,:])
#         else:
#             ax = fig.add_subplot(111)
#             ax.plot(x[0,:], x[1,:])
#     return x
#
# def Tao(data):
#     corr=np.correlate(data, data, mode="full")
#     corr=corr[int(len(corr)/2):len(corr)]
#     tau=0
#     j=0
#     while (corr[j]>0):
#         j=j+1
#     tau=j
#     return tau
#
# def fnn(data, maxm):
#     RT=15.0
#     AT=2
#     sigmay=np.std(data, ddof=1)
#     nyr=len(data)
#     m=maxm
#     EM=lagmat(data, maxlag=m-1)
#     EEM=np.asarray([EM[j,:] for j in range(m-1, EM.shape[0])])
#     embedm=maxm
#     for k in tqdm(range(AT,EEM.shape[1]+1)):
#         fnn1=[]
#         fnn2=[]
#         Ma=EEM[:,range(k)]
#         D=dist(Ma)
#         for i in range(1,EEM.shape[0]-m-k):
#             #print D.shape
#             #print(D[i,range(i-1)])
#             d=D[i,:]
#             pdnz=np.where(d>0)
#             dnz=d[pdnz]
#             Rm=np.min(dnz)
#             l=np.where(d==Rm)
#             l=l[0]
#             l=l[len(l)-1]
#             if l+m+k-1<nyr:
#                 fnn1.append(np.abs(data[i+m+k-1]-data[l+m+k-1])/Rm)
#                 fnn2.append(np.abs(data[i+m+k-1]-data[l+m+k-1])/sigmay)
#         Ind1=np.where(np.asarray(fnn1)>RT)
#         Ind2=np.where(np.asarray(fnn2)>AT)
#         if len(Ind1[0])/float(len(fnn1))<0.1 and len(Ind2[0])/float(len(fnn2))<0.1:
#             embedm=k
#             break
#     return embedm

def relation_baier_klein(x, y, z):
    return 1.76 - y*y - 0.1*z, x, y

def baier_klein(x0, y0, z0, N):
    x = [x0]
    y = [y0]
    z = [z0]

    for i in range(N - 1):
        xn, yn, zn = relation_baier_klein(x[i], y[i], z[i])
        x.append(xn)
        y.append(yn)
        z.append(zn)

    return x, y, z

def plot_baier_klein(res):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Baier Klein System")
    ax.scatter(res[0], res[1], res[2], color="red", lw=1)
    plt.show()

def lyapunov_test(series, eps):

    def d(series,i,j):
        return abs(series[i]-series[j])

    N=len(series)
    dlist=[[] for i in range(N)]
    n=0 #number of nearby pairs found
    for i in tqdm(range(N)):
        for j in range(i+1,N):
            if d(series,i,j) < eps:
                n+=1
                for k in range(min(N-i,N-j)):
                    dlist[k].append(log(d(series,i+k,j+k)))

    dk = []
    for i in range(len(dlist)):
        if len(dlist[i]):
            dk.append(sum(dlist[i])/len(dlist[i]))

    return [i for i in range(len(dlist))], dk


def ikeda_relation(z):
    complex_z = 0.4j - 6.j/(1 + abs(z)*abs(z))
    return 1 + 0.9*z*np.exp(complex_z)

def ikeda(z0, N):
    z = [z0]

    for i in range(N-1):
        z.append(ikeda_relation(z[i]))

    return z


class Exercice1:
    def question1():
        dt = 1e-3
        step = 100000
        x, y, z = BE1.lorenz_runge_kutta_4(dt, step, [1., 1., 1.], [10, 28, 8/3.0])
        BE1.plot_lorentz([x, y, z])

    def question2():
        LORENZ_MAP = LorenzMap()
        LORENZ_MAP_INITIAL_CONDITION = np.array([1,1,1])

        l = lyapunov_exponent(LORENZ_MAP, initial_condition=LORENZ_MAP_INITIAL_CONDITION, max_it=10000)
        print(max(l))

class Exercice2:
    def question1():
        x, y, z = baier_klein(0, 0, 0, 10000)
        plot_baier_klein([x, y, z])

    def question2():
        BAIER_KLEIN_MAP = BaierKleinMap(h0=0.01)
        BAIER_KLEIN_MAP_INITIAL_CONDITION = np.array([0,0,0])
        ls = []
        l = lyapunov_exponent(BAIER_KLEIN_MAP, initial_condition=BAIER_KLEIN_MAP_INITIAL_CONDITION, max_it=50)
        print(max(l))

class Exercice3:
    def question1():
        zn = ikeda(z0=0, N=5000)
        xn = [z.real for z in zn]
        xn = [(x - np.mean(xn))/np.std(xn) for x in xn]

        m=fnn(xn, 15)
        tau=Tao(xn)

        x = PhaseSpace(xn, m, tau)
        print(f'Dim Corrélation: {Dim_Corr(x)}')

    def question2():
        N = 5000
        zn = ikeda(z0=0, N=N)

        xn = [z.real for z in zn]
        xn = [(x - np.mean(xn))/np.std(xn) for x in xn]

        bruit = np.random.normal(loc=0, scale=0.02, size=N)
        xn = xn + bruit

        m=fnn(xn, 15)
        tau=Tao(xn)

        x = PhaseSpace(xn, m, tau)
        print(f'Dim Corrélation: {Dim_Corr(x)}')

    def question3():
        N = 5000
        zn = ikeda(z0=0, N=N)

        xn = [z.real for z in zn]
        xn = [(x - np.mean(xn))/np.std(xn) for x in xn]

        yn = [z.imag for z in zn]
        yn = [(y - np.mean(yn))/np.std(yn) for y in yn]

        bruit = np.random.normal(loc=0, scale=0.02, size=N)
        xn_bruit = xn + bruit
        yn_bruit = yn + bruit

        print(f'Dim Corrélation sans bruit: {Dim_Corr(np.asarray([xn, yn]))}')
        print(f'Dim Corrélation avec bruit: {Dim_Corr(np.asarray([xn_bruit, yn_bruit]))}')


if __name__ == '__main__':
    print('# Question 1\n')
    dt = 1e-3
    step = 100000
    x, y, z = BE1.lorenz_runge_kutta_4(dt, step, [1., 1., 1.], [10, 28, 8/3.0])
    # BE1.plot_lorentz([x, y, z])

    print('# Question 2\n')
    from lyapunov import lyapunov_rosenstein

    x = np.array(x)
    d = lyapunov_rosenstein(x, m=10, time_delay=10, mean_period=1, max_iter=100)
    print(d)
