import numpy as np
import matplotlib.pyplot as plt

from functions import utils, lorentz, plongement


def Exercice1():
    print('# Question 1 & 2 \n')

    N = 1024
    x = np.random.normal(loc=0, scale=1, size=(N + 14))
    y = np.asarray(utils.filtre_mobile(x))

    for data, label in [(x, 'xn'), (y, 'yn')]:
        freqs, ps, idx = utils.DSP(data, 0.01)
        max_ps = max(ps)
        plt.plot(freqs[idx], [p/max_ps for p in ps[idx]], label=label)

    plt.legend()
    plt.show()

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
    plt.legend()
    plt.show()


def Exercice3():
    print('# Question 1\n')

    dt = 1e-3
    step = 100000
    res = lorentz.lorenz_runge_kutta_4(dt, step, [1., 1., 1.], [10, 28, 8/3.0])
    lorentz.plot_lorentz(res)

    print('# Question 2\n')
    plongement.animation_plongement_2d(res[0], tau_max=200, tau_pas=10)

    print('# Question 3\n')
    
