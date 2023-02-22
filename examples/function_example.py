"""
This example is meant to demonstrate how you can map complex
functions using a single input and single output with polynomial
synaptic weights
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim
from adahessian_torch.optim_adahessian import Adahessian
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from high_order_layers_torch.layers import *


class simple_func:
    def __init__(self):
        self.factor = 1.5 * 3.14159
        self.offset = 0.25

    def __call__(self, x):
        return 0.5 * torch.cos(self.factor * 1.0 / (abs(x) + self.offset))


xTest = np.arange(1000) / 500.0 - 1.0
xTest = torch.stack([torch.tensor(val) for val in xTest])

xTest = xTest.view(-1, 1)
yTest = simple_func()(xTest)
yTest = yTest.view(-1, 1)


class FunctionDataset(Dataset):
    """
    Loader for reading in a local dataset
    """

    def __init__(self, transform=None):
        self.x = (2.0 * torch.rand(1000) - 1.0).view(-1, 1)
        self.y = simple_func()(self.x)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x.clone().detach()[idx], self.y.clone().detach()[idx]


class PolynomialFunctionApproximation(LightningModule):
    """
    Simple network consisting of on input and one output
    and no hidden layers.
    """

    def __init__(
        self, n, segments=2, function=True, periodicity=None, opt: str = "adam"
    ):
        super().__init__()
        self.automatic_optimization = False
        self.optimizer = opt

        self.layer = high_order_fc_layers(
            layer_type=function,
            n=n,
            in_features=1,
            out_features=1,
            segments=segments,
            length=2.0,
            periodicity=periodicity,
        )

    def forward(self, x):
        return self.layer(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, y = batch
        x.requires_grad_(True)
        y.requires_grad_(True)
        y_hat = self(x)
        y_hat.requires_grad_(True)
        #print('y_hat', y_hat)

        loss = F.mse_loss(y_hat, y)

        opt.zero_grad()
        #self.manual_backward(loss, create_graph=False)
        grad = torch.autograd.grad(loss, self.layer.parameters(), torch.ones_like(loss),create_graph=True,allow_unused=True)
        #print('self.params', list(self.layer.parameters()))
        #print('grad', grad)
        for index, param in enumerate(self.layer.parameters()) :
            param.grad = grad[index]

        opt.step()
        self.log(f"loss", loss, prog_bar=True)

        return {"loss": loss}

    def train_dataloader(self):
        return DataLoader(FunctionDataset(), batch_size=256)

    def configure_optimizers(self):
        if self.optimizer == "adahessian":
            return Adahessian(
                self.layer.parameters(),
                lr=1.0,
                betas=(0.9, 0.999),
                eps=1e-4,
                weight_decay=0.0,
                hessian_power=1.0,
            )
        elif self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=0.001)
        elif self.optimizer == "lbfgs":
            return torch.optim.LBFGS(
                self.parameters(), lr=1, max_iter=20, history_size=100
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer} not recognized")


modelSetL = [
    {"name": "Relu 2", "n": 2},
    {"name": "Relu 3", "n": 8},
    {"name": "Relu 4", "n": 16},
]

modelSetProd = [
    {"name": "Product 2", "n": 2},
    {"name": "Product 3", "n": 8},
    {"name": "Product 4", "n": 16},
]

modelSetD = [
    {"name": "Discontinuous", "n": 2},
    # {'name': 'Discontinuous 2', 'order' : 2},
    {"name": "Discontinuous", "n": 4},
    # {'name': 'Discontinuous 4', 'order' : 4},
    {"name": "Discontinuous", "n": 6},
]

modelSetC = [
    {"name": "Continuous", "n": 2},
    # {'name': 'Continuous 2', 'order' : 2},
    {"name": "Continuous", "n": 4},
    # {'name': 'Continuous 4', 'order' : 4},
    {"name": "Continuous", "n": 6},
]

modelSetP = [
    {"name": "Polynomial", "n": 30},
]

modelSetF = [
    {"name": "Fourier", "n": 10},
    # {'name': 'Continuous 2', 'order' : 2},
    {"name": "Fourier", "n": 20},
    # {'name': 'Continuous 4', 'order' : 4},
    {"name": "Fourier", "n": 30},
]

colorIndex = ["red", "green", "blue", "purple", "black"]
symbol = ["+", "x", "o", "v", "."]


def plot_approximation(
    function,
    model_set,
    segments,
    epochs,
    gpus=0,
    periodicity=None,
    plot_result=True,
    opt="adahessian",
):
    for i in range(0, len(model_set)):

        trainer = Trainer(max_epochs=epochs, gpus=gpus)

        model = PolynomialFunctionApproximation(
            n=model_set[i]["n"],
            segments=segments,
            function=function,
            periodicity=periodicity,
            opt=opt,
        )

        trainer.fit(model)
        predictions = model(xTest.float())

        if plot_result is True:
            plt.scatter(
                xTest.data.numpy(),
                predictions.flatten().data.numpy(),
                c=colorIndex[i],
                marker=symbol[i],
                label=f"{model_set[i]['name']} {model_set[i]['n']}",
            )

    if plot_result is True:
        plt.plot(
            xTest.data.numpy(), yTest.data.numpy(), "-", label="actual", color="black"
        )
        plt.title("Piecewise Polynomial Function Approximation")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()


def plot_results(
    epochs: int = 20, segments: int = 5, plot: bool = True, opt: str = "adahessian"
):

    """
    plt.figure(0)
    plot_approximation("standard", modelSetL, 1, epochs, gpus=0)
    plt.title('Relu Function Approximation')
    """
    """
    plt.figure(0)
    plot_approximation("product", modelSetProd, 1, epochs, gpus=0)
    """

    data = [
        {
            "title": "Polynomial function approximation",
            "layer": "polynomial",
            "model_set": modelSetP,
        },
    ]

    for index, element in enumerate(data):
        if plot is True:
            plt.figure(index)
        plot_approximation(
            function=element["layer"],
            model_set=element["model_set"],
            segments=5,
            epochs=epochs,
            gpus=0,
            periodicity=2,
            opt=opt,
        )

        if plot is True:
            plt.title("Piecewise Discontinuous Function Approximation")

    if plot is True:
        plt.show()


if __name__ == "__main__":
    plot_results(opt="adahessian")
