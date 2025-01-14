"""
This example is meant to demonstrate how you can map complex
functions using a single input and single output with polynomial
synaptic weights
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim
from high_order_layers_optimizers.optim_adahessian import Adahessian
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf
import hydra
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

    def __init__(self, cfg: DictConfig, function=True):
        super().__init__()
        self._cfg = cfg
        self.automatic_optimization = False
        self.optimizer = cfg.optimizer.name

        self.layer = high_order_fc_layers(
            layer_type=function,
            n=cfg.n,
            in_features=1,
            out_features=1,
            segments=cfg.segments,
            length=2.0,
            periodicity=None,
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

        loss = F.mse_loss(y_hat, y)

        opt.zero_grad(set_to_none=True)
        # self.manual_backward(loss, create_graph=False)
        grad = torch.autograd.grad(
            loss,
            self.layer.parameters(),
            torch.ones_like(loss),
            create_graph=True,
            allow_unused=True,
        )

        for index, param in enumerate(self.layer.parameters()):
            param.grad = grad[index]

        opt.step()
        self.log(f"loss", loss, prog_bar=True)

        #opt.zero_grad(set_to_none=True)
        #for param in self.layer.parameters() :
        #    del param.grad

        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0)))

        #return {"loss": loss}

    def train_dataloader(self):
        return DataLoader(FunctionDataset(), batch_size=256)

    def configure_optimizers(self):
        if self.optimizer == "adahessian":
            return Adahessian(
                self.layer.parameters(),
                lr=self._cfg.optimizer.lr,
                betas=(0.9, 0.999),
                eps=1e-4,
                weight_decay=0.0,
                hessian_power=1.0,
            )
        elif self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self._cfg.optimizer.lr)
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


def plot_approximation(function, cfg):
    print("inside here")
    for i in range(1):
        trainer = Trainer(max_epochs=cfg.epochs, gpus=cfg.gpus)

        model = PolynomialFunctionApproximation(
            cfg=cfg,
            function=function,
        )

        trainer.fit(model)
        predictions = model(xTest.float())

        if cfg.plot is True:
            plt.scatter(
                xTest.data.numpy(),
                predictions.flatten().data.numpy(),
                # c=colorIndex[i],
                # marker=symbol[i],
                # label=f"{model_set[i]['name']} {model_set[i]['n']}",
            )

    if cfg.plot is True:
        plt.plot(
            xTest.data.numpy(), yTest.data.numpy(), "-", label="actual", color="black"
        )
        plt.title("Function Approximation")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()


def plot_results(cfg: DictConfig):
    for index in range(1):
        print("index", index)
        if cfg.plot is True:
            plt.figure(index)
        plot_approximation(function=cfg.layer_type, cfg=cfg)

        if cfg.plot is True:
            plt.title("Piecewise Discontinuous Function Approximation")

    if cfg.plot is True:
        plt.show()


@hydra.main(config_path="../config", config_name="function_example")
def run(cfg: DictConfig):
    plot_results(cfg)


if __name__ == "__main__":
    run()
