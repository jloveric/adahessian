import torch


def rastrigin(x: torch.Tensor, A: float = 10):
    return A + sum((x**2 - A * torch.cos(2 * torch.pi * x)))
