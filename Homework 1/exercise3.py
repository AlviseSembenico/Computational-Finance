from base import *


class GeometricBrownianMotion:

    W: np.ndarray
    time: np.ndarray

    def __init__(
        self,
        t: float,
        steps: int = 1000,
        paths: int = 1,
        beta: float = 1,
        sigma: float = 1,
        x_0: float = 0,
        r: float = 1,
    ):
        Z = np.random.normal(0.0, 1.0, [paths, steps])
        X = np.zeros([paths, steps + 1])
        X[:, 0] = x_0

        dt = t / steps
        for i in range(steps):
            if paths > 1:
                Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])

            X[:, i + 1] = (
                X[:, i]
                + (r * X[:, i]) * dt
                + sigma * X[:, i] * np.power(dt, 0.5) * Z[:, i]
            )

        self.X = X
        self.time = np.linspace(0, t, steps + 1)


# Set parameters
r = 0.06
t = 7
paths = 1000
steps = 1000


def main():

    # Simulate the paths
    X = GeometricBrownianMotion(
        t=t, paths=paths, steps=steps, beta=0.04, sigma=0.38, x_0=4.0, r=r
    )
    Y = GeometricBrownianMotion(
        t=t, paths=paths, steps=steps, beta=0.1, sigma=0.15, x_0=1.0, r=r
    )
    M = np.exp(X.time * r)

    # plot the process X
    plt.figure()
    plt.plot(X.time, X.X.T)
    plt.title("Process X")
    plt.xlabel("t")
    plt.ylabel("X(t)")
    plt.show()

    # Plot the combined process
    process = 0.5 * X.X - 0.5 * Y.X
    plt.figure()
    plt.plot(X.time, process.T)
    equation = r"$\frac{1}{2}X(t) - \frac{1}{2}Y(t)$"
    plt.title(equation)
    plt.xlabel("t")
    plt.ylabel(equation)
    plt.show()

    # compute the expected value V(t)
    result_v = []
    k_space = np.linspace(0, 10, num=200)
    for k in k_space:
        K = np.zeros(steps + 1).reshape(1, steps + 1) + k
        v = 1 / M * np.maximum(0.5 * X.X - 0.5 * Y.X, K)
        result_v.append(np.average(v[:, -1]))

    plt.figure(1)
    plt.plot(np.linspace(0, 10, num=200), result_v)
    plt.title("V(t)")
    plt.xlabel("K")
    plt.ylabel("V(7y, K)")
    plt.show()


if __name__ == "__main__":
    main()
