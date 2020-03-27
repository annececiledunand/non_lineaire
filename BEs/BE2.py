import numpy as np
from tqdm import tqdm

from functions import attractors, correlation_dim, lyapunov


def Exercice1():
    ######## Question 1 ########
    print('# Question 1\n')

    # dt, step : parametres pour faire apparaitre l'attracteur, génère trop de points
    # pour les traitements suivants, mais je n'ai pas trouvé d'autres combinaisons viables
    # sans doute fait il changer les conditions initales.
    # la plupart des calculs liés à Lorentz ont été réalisé une seule fois, car
    # trop couteux de les lancer à chaque fois
    dt = 1e-3
    step = 100000
    params = [10, 28, 8/3.0]
    x, y, z = attractors.lorenz_runge_kutta_4(dt, step, y0s=[1., 1., 1.], params=params)
    attractors.plot_lorentz([x, y, z], params)


    ######## Question 2 ########
    print('# Question 2\n')

    # trop long a calculer, même sur la dernière moitié du signal
    # d = correlation_dim.find_d_embedding(x, 10)
    # tau = correlation_dim.find_tau(x)
    # exponent = lyapunov.lyapunov_rosenstein(np.array(x), d, time_delay=tau, mean_period=1, max_iter=100)

    d = 4
    tau = 50
    exponent = [0.6053546813188815]
    print(f'Lyapunov Exposant for d={d}, tau={tau}: ', max(exponent))
    print(f'Lyapunov Exposant attendu : 0.84')


    ######## Question 3 ########
    print('\n# Question 3 \n')
    corr_dim = 1.9455389141366316
    # corr_dim = correlation_dim.correlation_dim(x, tau=tau, d=d)
    print(f'Dimension de correlation : {corr_dim}')
    print(f'Dimension attendue : 2.06')


def Exercice2():
    ######## Question 1 ########
    print('# Question 1\n')
    x, y, z = attractors.baier_klein(x0=0, y0=0, z0=0, N=10000)
    attractors.plot_baier_klein([x, y, z])


    ######## Question 2 ########
    print('# Question 2 (3 minutes de traitement)\n')
    x_square = [xi*xi for xi in x]
    d = correlation_dim.find_d_embedding(x_square, 10)
    print(f'Found embedding dimension d={d}\n')

    for N in range(1000, 11000, 1000):
        x, _, _ = attractors.baier_klein(0, 0, 0, N)
        x_square = [xi*xi for xi in x]

        exponent = lyapunov.lyapunov_rosenstein(np.array(x_square), d, time_delay=0, mean_period=1, max_iter=100)
        print(f'Lyapunov exponent for N={N} : {max(exponent)}')


    ######## Question 3 ########
    print('\n# Question 3 (10 minutes de traitement)\n')
    for scale in [0.01, 0.05, 0.1]:
        print(f'\nBruit à {scale*100} % du niveau du signal.\n')

        for N in range(1000, 11000, 1000):
            x, _, _ = attractors.baier_klein(0, 0, 0, N)

            x_square = [xi*xi for xi in x]
            bruit = np.random.normal(loc=0, scale=scale*np.std(x_square), size=N)
            data = np.array(x_square) + bruit

            exponent = lyapunov.lyapunov_rosenstein(data, d, time_delay=0, mean_period=1, max_iter=100)
            print(f'Lyapunov exponent avec bruit {scale*100}% for N={N} : {max(exponent)}')


def Exercice3():
    ######## Question 1 ########
    print(f'# Question 1\n')
    N = 5000
    zn = attractors.ikeda(z0=0, N=N)
    xn = [z.real for z in zn]
    xn = [(x - np.mean(xn))/np.std(xn) for x in xn]  # normalisation du signal

    d = correlation_dim.find_d_embedding(xn, 10)
    tau = correlation_dim.find_tau(xn)

    dim_corr = correlation_dim.correlation_dim(xn, tau, d)
    print(f'\nDim corrélation Ikeda Reel pour N={N}, tau={tau}, d={d} : {dim_corr}')


    ######## Question 2 ########
    print(f'# Question 2\n')
    bruit = np.random.normal(loc=0, scale=0.02, size=N)
    xn_bruit = xn + bruit

    dim_corr = correlation_dim.correlation_dim(xn_bruit, tau, d)
    print(f'\nDim Corrélation Ikeda Reel bruité pour N={N}, tau={tau}, d={d} : {dim_corr}')


    ######## Question 3 ########
    print(f'# Question 3\n')
    yn = [z.imag for z in zn]
    yn = [(y - np.mean(yn))/np.std(yn) for y in yn]

    bruit = np.random.normal(loc=0, scale=0.02, size=N)
    yn_bruit = yn + bruit

    dim_corr = correlation_dim.correlation_dim(yn, tau, d)
    dim_corr_bruite = correlation_dim.correlation_dim(yn_bruit, tau, d)
    print(f'\nDim Corrélation Ikeda Imaginaire pour N={N}, tau={tau}, d={d} : {dim_corr}')
    print(f'\nDim Corrélation Ikeda Imaginaire bruité pour N={N}, tau={tau}, d={d} : {dim_corr_bruite}\n')
