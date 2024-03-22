import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

N = norm.cdf


class GeometricBrownianMotion:

    W: np.ndarray
    time: np.ndarray

    def __init__(
        self,
        Z,
        t: float,
        steps: int = 1000,
        paths: int = 1,
        beta: float = 1,
        sigma: float = 1,
        x_0: float = 0,
        r: float = 1,
        normalization: bool = True,
    ):
        X = np.zeros([paths, steps + 1])
        X[:, 0] = x_0

        dt = t / steps
        for i in range(steps):
            W = Z[:, i]
            if paths > 1 and normalization:
                W = (W - np.mean(W)) / np.std(W)

            X[:, i + 1] = (
                X[:, i] + (r * X[:, i]) * dt + sigma * X[:, i] * np.power(dt, 0.5) * W
            )

        self.X = X
        self.time = np.linspace(0, t, steps + 1)


def BS_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r * T) * N(d2)


r = 0.05
t = 1
x_0 = 10
k = 10
sigma = 0.5
steps = 1000


def main():

    paths = np.linspace(0, 1000, num=200, dtype=int)

    normalized = []
    unnormalized = []
    Z = np.random.normal(0.0, 1.0, [paths.max(), steps])
    Y_normalized = GeometricBrownianMotion(
        Z=Z,
        t=t,
        paths=paths.max(),
        steps=steps,
        beta=0.1,
        sigma=sigma,
        x_0=x_0,
        r=r,
        normalization=True,
    )
    Y_unnormalized = GeometricBrownianMotion(
        Z=Z,
        t=t,
        paths=paths.max(),
        steps=steps,
        beta=0.1,
        sigma=sigma,
        x_0=x_0,
        r=r,
        normalization=False,
    )

    for path in paths:
        for Y, vector in zip(
            [Y_normalized, Y_unnormalized], [normalized, unnormalized]
        ):
            X = Y.X[:path]
            mean = np.maximum(X[:, -1] - k, 0).mean()
            vector.append(mean)

    true_value = BS_call(x_0, k, t, r, sigma)

    plt.figure()
    plt.plot(paths, normalized, label="Mean of normalized")
    plt.plot(paths, unnormalized, label="Mean of unnormalized")
    plt.plot(paths, [true_value] * len(paths), label="True value")
    plt.legend()


if __name__ == "__main__":
    main()
    plt.show()
