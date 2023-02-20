"""
Modified from this gist
https://gist.github.com/miku/fca6afe05d65302f14c2b6f5242458d6
"""

import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    X = torch.linspace(-4, 4, 200)
    Y = torch.linspace(-4, 4, 200)

    X, Y = torch.meshgrid(X, Y)

    Z = rastrigin(X, Y, A=10)

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False
    )
    plt.savefig("rastrigin.png")
