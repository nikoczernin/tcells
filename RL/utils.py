import matplotlib.pyplot as plt
import numpy as np



def plot_line_graph(*values_lists, title="Line Graph", xlabel="X-axis", ylabel="Y-axis", labels=None):
    for i, values in enumerate(values_lists):
        if labels:
            plt.plot(values, label=labels[i])
        else:
            plt.plot(values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if labels:
        plt.legend()
    plt.show()


def plot_blockwise_mean_rewards_line_graph(rewards: list, k=80, j=8,
                                           title="Line Graph", xlabel="X-axis", ylabel="Y-axis", labels=None):
    print(f"Training rewards: [{', '.join(str(x) for x in rewards[:j])}, ..., {', '.join(str(x) for x in rewards[-j:])}]")
    n = len(rewards)
    step_size = n//k
    means = [np.mean(rewards[i: i + step_size]) for i in range(0, n, step_size)]
    plt.plot(means, label=labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if labels:
        plt.legend()
    plt.xticks(np.arange(0, k+1, n//step_size//10), np.arange(0, k+1, n//step_size//10)*step_size)
    plt.show()


def rational_function(x):
    return x / (1 + x)

def positive_mountain(x, flatness=60, flatness_start=3):
    return -x**2/flatness + x/flatness_start

def softmax(x):
    # subtract max for numerical stability
    e = np.exp(x - np.max(x))
    return e / e.sum()

def stable_softmax(x):
    # if any +inf, give all mass to those entries
    is_inf = np.isposinf(x)
    if np.any(is_inf):
        probs = np.zeros_like(x, dtype=float)
        # split evenly among all +inf entries
        probs[is_inf] = 1.0 / is_inf.sum()
        return probs

    # otherwise proceed normally
    x_shift = x - np.max(x)
    e = np.exp(x_shift)
    return e / e.sum()