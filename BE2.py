import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import math

from functions import lorentz, correlation_dim
from functions.BE2_LEGACY import fnn, Tao
from functions.lyapunov import phase_space_reconstruction, lyapunov_rosenstein


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


def Exercice1():
    print('# Question 1\n')
    dt = 1e-3
    step = 100000
    x, y, z = lorentz.lorenz_runge_kutta_4(dt, step, [1., 1., 1.], [10, 28, 8/3.0])
    lorentz.plot_lorentz([x, y, z])

    # print('# Question 2\n')
    # x = np.array(x)
    # d = lyapunov_rosenstein(x, m=10, time_delay=10, mean_period=1, max_iter=100)
    # print('Lyapunov Exposant : ', max(d))

    print('# Question 3\n')
    correlation_dim.correlation_dim(x, tau=5, d=2)


def Exercice1_LEGACY():
    print('# Question 1\n')
    dt = 1e-3
    step = 100000
    x, y, z = BE1.lorenz_runge_kutta_4(dt, step, [1., 1., 1.], [10, 28, 8/3.0])
    BE1.plot_lorentz([x, y, z])

    print('# Question 2\n')
    LORENZ_MAP = LorenzMap()
    LORENZ_MAP_INITIAL_CONDITION = np.array([1,1,1])

    l = lyapunov_exponent(LORENZ_MAP, initial_condition=LORENZ_MAP_INITIAL_CONDITION, max_it=10000)
    print(max(l))


def Exercice2():
    print('# Question 1\n')
    x, y, z = baier_klein(0, 0, 0, 10000)
    plot_baier_klein([x, y, z])


def Exercice3_LEGACY():
    print('# Question 1\n')
    zn = ikeda(z0=0, N=5000)
    xn = [z.real for z in zn]
    xn = [(x - np.mean(xn))/np.std(xn) for x in xn]

    m=fnn(xn, 15)
    tau=Tao(xn)

    x = PhaseSpace(xn, m, tau)
    print(f'Dim Corrélation: {Dim_Corr(x)}')

    print('# Question 2\n')
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



def Exercice3():
    print('# Question 1\n')
    zn = ikeda(z0=0, N=5000)
    xn = [z.real for z in zn]
    xn = [(x - np.mean(xn))/np.std(xn) for x in xn]

    xn = np.array(xn)
    x = phase_space_reconstruction(xn, m=15, time_delay=1)
    print(correlation_dim.correlation_dim(x, d=2, dt=0.2, nb_iter=100))
    print(f'Dim Corrélation: {Dim_Corr(x)}')

    print('# Question 2\n')
    N = 5000
    zn = ikeda(z0=0, N=N)

    xn = [z.real for z in zn]
    xn = [(x - np.mean(xn))/np.std(xn) for x in xn]

    bruit = np.random.normal(loc=0, scale=0.02, size=N)
    xn = xn + bruit

    m = fnn(xn, 15)
    time_delay = Tao(xn)

    x = phase_space_reconstruction(xn, m, time_delay)
    print(f'Dim Corrélation: {Dim_Corr(x)}')


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
