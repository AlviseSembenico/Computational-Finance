import math
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy import optimize
from scipy.stats import norm
from tqdm import tqdm

N = norm.cdf


def f(x):
    return (math.e**x + math.e ** (-x)) / 2 - 2 * x


def df_dx(x):
    return (math.e**x - math.e ** (-x)) / 2 - 2


def plot_f():
    x = np.linspace(0, 3, 100)
    y = f(x)

    plt.figure()
    plt.plot(x, y, label="f(x)")
    plt.plot(x, [0] * len(x), label="y=0")
    plt.legend()
    plt.savefig("3_f_x.png")


def compute_root(initials):
    result = {
        "newton": ([], []),
        "bisect": ([], []),
        "brenth": ([], []),
    }
    plt.figure()
    for i in range(1, 10):
        for optimizer_name, interval, additional_params in zip(
            result.keys(), initials, [{"fprime": df_dx}, None, None]
        ):
            args = {"rtol": 1e-6, "full_output": True, "disp": False}
            args["maxiter"] = i
            if additional_params:
                args = {**args, **additional_params}
            optimizer = getattr(optimize, optimizer_name)
            x, _ = optimizer(f, *interval, **args)
            result[optimizer_name][0].append(x)
            result[optimizer_name][1].append(f(x))

    plt.figure()
    for optimizer_name, values in result.items():
        plt.plot(
            np.linspace(0, 10, num=len(values[0])), values[0], label=optimizer_name
        )
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Root")
    plt.savefig("3_optimizers1.png")


def BS_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r * T) * N(d2)


def objective_function(S, T, r, K, option_price, sigma):
    return BS_call(S, K, T, r, sigma) - option_price


def compute_implied_volatility(df, S, T, r):
    computed_volatility = []
    for index, row in tqdm(df.iterrows()):
        K = row["strike"]
        option_price = (row["ask"] + row["bid"]) / 2

        # initial guess for the volatility
        # otherwise it can be that it does not converge
        sigma_grid = np.linspace(0, 2, 200)
        sigma_start = sigma_grid[
            np.argmin(
                [
                    np.abs(
                        objective_function(
                            S=S, T=T, r=r, K=K, option_price=option_price, sigma=sigma
                        )
                    )
                    for sigma in sigma_grid
                ]
            )
        ]

        implied_volatility = optimize.newton(
            lambda sigma: objective_function(
                S=S, T=T, r=r, K=K, option_price=option_price, sigma=sigma
            ),
            sigma_start,
            maxiter=1000,
        )
        computed_volatility.append(implied_volatility)

    df["computed_volatility"] = computed_volatility
    plt.figure()
    plt.plot(df.strike, df.computed_volatility, label="computed")
    plt.plot(df.strike, df.impliedVolatility, label="implied")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.legend()
    plt.savefig("3_implied_volatility.png")


def main():

    compute_root([(0,), (0, 1), (0, 1)])
    compute_root([(2,), (2, 3), (2, 3)])

    ### Part 2
    # read data previously downloaded
    option_date = "2025-03-21"
    df = pandas.read_csv("GOOG_options.csv")

    # setting the data
    expire_date = datetime.strptime(option_date, "%Y-%m-%d")
    T = (expire_date - datetime.now()).days / 365
    # risk free rate
    r = 0.02
    # stock price at the time when the data was downloaded
    S = 148.8006
    compute_implied_volatility(df, S, r, T)


if __name__ == "__main__":
    main()
