from base import *


class Custom:

    W: np.ndarray
    time: np.ndarray
    I1: np.ndarray
    I2: np.ndarray

    def __init__(self, t: float, steps: int = 1000, paths: int = 1):
        Z = np.random.normal(0.0, 1.0, [paths, steps])
        W = np.zeros([paths, steps + 1])

        I1 = np.zeros([paths, steps + 1])
        I2 = np.zeros([paths, steps + 1])

        dt = t / steps
        for i in range(steps):
            if paths > 1:
                Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
            W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
            I2[:, i + 1] = I2[:, i] + t * (1 - i / steps) * (W[:, i + 1] - W[:, i])
            I1[:, i + 1] = I1[:, i] + W[:, i] * dt

        self.W = W
        self.I1 = I1
        self.I2 = I2
        self.time = np.linspace(0, t, steps + 1)

    def plot(self):

        plt.figure(1)
        plt.plot(self.time, self.I1.squeeze(), label="I1")
        plt.plot(self.time, self.I2.squeeze(), label="I2")
        plt.xlabel("t")
        plt.ylabel("Integral value")
        plt.legend()
        plt.show()


def main():
    Custom(t=5, paths=1, steps=int(1e4)).plot()


if __name__ == "__main__":
    main()
