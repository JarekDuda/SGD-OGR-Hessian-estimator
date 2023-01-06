from sgd_ogr.adam import SimplifiedTorchAdam
from sgd_ogr.dogr import dOGR
from sgd_ogr.cdogr import cdOGR

import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_optimizer(optim, n_steps, title=None, **params):
    def f(w):
        x = torch.tensor([0.2, 2], dtype=torch.float)
        return torch.sum(x * w ** 2)

    w = torch.tensor([-6, 2], dtype=torch.float, requires_grad=True)

    optimizer = optim([w], **params)

    history = [w.clone().detach().numpy()]

    for i in range(n_steps):
        optimizer.zero_grad()

        loss = f(w)
        loss.backward()
        optimizer.step()
        history.append(w.clone().detach().numpy())

    delta = 0.01
    x = np.arange(-7.0, 7.0, delta)
    y = np.arange(-4.0, 4.0, delta)
    X, Y = np.meshgrid(x, y)

    Z = 0.2 * X ** 2 + 2 * Y ** 2

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.contour(X, Y, Z, 20)

    h = np.array(history)

    ax.plot(h[:, 0], h[:, 1], 'x-')

    if title is not None:
        ax.set_title(title)

    plt.show()


if __name__ == '__main__':
    visualize_optimizer(SimplifiedTorchAdam, 20, 'Simplified Torch Adam', lr=0.35)
    visualize_optimizer(torch.optim.SGD, 20, 'Torch SGD', lr=0.35)
    visualize_optimizer(dOGR, 40, 'dOGR', lr=0.05, beta=0.5, div=1.5, cut=5.0)
    visualize_optimizer(cdOGR, 20, 'cdOGR', lr=0.004, beta=0.6, div=1.1, cut=0.2)

