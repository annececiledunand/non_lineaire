import matplotlib.pyplot as plt

from functions import attractors, correlation_dim, plongement


def Exercice1():
    ######## Question 1 ########
    print('# Question 1\n')

    a, b = 1.4, 0.3
    N = 15000
    x, y = attractors.henon(0, 0, a, b, N)
    attractors.plot_henon(x, y, [a, b, N])


    ######## Question 2 ########
    print('# Question 2\n')

    tau = correlation_dim.find_tau(x)
    d = correlation_dim.find_d_embedding(x, 10)
    corr_dim = correlation_dim.correlation_dim(x, tau, d)

    print('\nAttendue Correlation dim : 1.22')
    print(f'Correlation dim for tau={tau}, d={d} : {corr_dim}\n')


    ######## Question 3 ########
    print('# Question 3\n')

    entropy = correlation_dim.entropy(x, d=d)

    print('Attendue entropie : 0.325')
    print(f'Entropy : {entropy}')
