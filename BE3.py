import matplotlib.pyplot as plt

# xn+1 = 1 - axn^2 +yn
#  yn = bnx

a = 1.4
b = 0.3
N = 15000


def henon_rec(x, y, a, b):
    return 1 - a*x*x + y, b*x


def henon(x0, y0, a, b, N):
    x, y = [x0], [y0]
    for i in range(N):
        xn, yn = henon_rec(x[i], y[i], a, b)
        x.append(xn)
        y.append(yn)

    return x, y


def plot_henon(x, y):
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Henon attractor")
    ax.scatter(x, y, color="red", s=0.4)
    plt.show()


# question 1
x, y = henon(0, 0, 1.4, 0.3, 15000)
plot_henon(x, y)
