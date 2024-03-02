from base import *


def analitic_variance(t: int = 10, steps: int = 1000):
    X = np.linspace(0, t, num=steps)
    var = X * (1 + (X / t**2) * (t - X) - 2 / t * np.minimum(X, t - X))
    return (X, var)


class BrownianMotion:

    W: np.ndarray
    time: np.ndarray
    var: np.ndarray
    analitic_variance: np.ndarray

    def __init__(self, t: float, steps: int = 1000, paths: int = 1):
        Z = np.random.normal(0.0, 1.0, [paths, steps])
        W = np.zeros([paths, steps + 1])

        dt = t / steps
        for i in range(steps):
            if paths > 1:
                Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
            W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]

        self.W = W
        X = np.zeros([paths, steps + 1])
        for i in range(steps + 1):
            X[:, i] = W[:, i] - (i / (steps)) * W[:, steps - i]

        self.X = X
        self.var = np.var(X, axis=0)
        self.time = np.linspace(0, t, steps + 1)
        _, self.analitic_variance = analitic_variance(t, steps + 1)

    def plot(self, paths: bool = True):

        plt.figure(1)
        if paths:
            plt.plot(self.time, np.transpose(self.W))
        plt.plot(self.time, self.analitic_variance, label="Analitic variance")
        plt.plot(self.time, self.var, label="Empirical variance")
        plt.xlabel("t")
        plt.ylabel("Var[X]")
        plt.title("Variance of X(t)")
        plt.legend()
        plt.show()


def main():
    BrownianMotion(t=10, paths=1000).plot(paths=False)
    bm = BrownianMotion(t=10, paths=50)
    plt.figure()
    plt.plot(bm.time, bm.X.T)
    plt.title("Wiener process")
    plt.xlabel("t")
    plt.ylabel("X(t)")
    plt.show()


if __name__ == "__main__":
    main()
