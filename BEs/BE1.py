import numpy as np
import matplotlib.pyplot as plt

from functions import utils, attractors, plongement, correlation_dim


def Exercice1():
    ######## Question 1 & 2 ########
    print('# Question 1 & 2 \n')

    N = 1024
    x = np.random.normal(loc=0, scale=1, size=(N + 14))
    y = np.asarray(utils.filtre_mobile(x))

    for data, label in [(x, 'xn'), (y, 'yn')]:
        freqs, ps, idx = utils.DSP(data, 0.01)
        max_ps = max(ps)
        plt.plot(freqs[idx], [p/max_ps for p in ps[idx]], label=label)

    plt.legend()
    plt.title(f'DSP de xn et yn (filtre moyenne mobile) pour N={N}')
    plt.show()


    ######## Question 3 ########
    print('# Question 3 \n')
    N = 4096

    xn = np.random.normal(loc=0, scale=1, size=N)
    yn = utils.processus_ulam(N, y0=0.1, rec_relation=lambda x: 1 - 2*x*x)
    sn = utils.fct_observation_sn(yn)

    for data, label in [(xn, 'xn'), (yn, 'yn'), (sn, 'sn')]:
        print(f'Valeurs pour {label} :')
        print(f'\tMoyenne  : {np.mean(data)}')
        print(f'\tVariance : {np.var(data)}')

        freqs, ps, idx = utils.DSP(data, 0.01)
        max_ps = max(ps)
        plt.plot(freqs[idx], [p/max_ps for p in ps[idx]], label=label)

    # On a égalité parfaite des DSP de xn et sn
    plt.title(f'DSP de xn, yn (Ulam processus) et sn (observation) pour N={N}')
    plt.legend()
    plt.show()


def Exercice2():
    pass


def Exercice3():
    ######## Question 1 ########
    print('# Question 1\n')

    dt = 1e-3
    step = 100000
    params = [10, 28, 8/3.0]
    x, y, z = attractors.lorenz_runge_kutta_4(dt, step, y0s=[1., 1., 1.], params=params)
    attractors.plot_lorentz([x, y, z], params)


    ######## Question 2 ########
    print('# Question 2\n')
    plongement.animation_plongement_2d(x, tau_max=200, tau_pas=10)


    ######## Question 3 ########
    print('# Question 3\n')

    # temps de calcul extremement long, on trouve d=2 avec la fonction suivante :
    # d = correlation_dim.find_d_embedding(x[50000:], 10)
    tau = correlation_dim.find_tau(x)
    d = 2

    # il y a clairement une erreur dans l'algorithme, pour tau, on utilise tau = 100
    print(f'Tau found : {tau}, Tau used : 100')
    print(f'Embedding dimension found : {d}')

    tau = 100
    X = plongement.plongement(x, d=d, tau=tau)
    plongement.plot_plongement_2d(X, tau)


    ######## Question 4 ########
    print('\n# Question 4\n')
    epsilon = 1e-2
    utils.plot_section_poincare(x, y, z, epsilon)
