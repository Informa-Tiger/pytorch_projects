import typing
from typing import Any, Union, Optional, List, Dict

import math
import numpy as np

import matplotlib.pyplot as plt

import torch
from torch import Tensor, ones
from torch.nn import Module, MSELoss, ModuleList, Sigmoid, ReLU, Identity, Tanh, LeakyReLU
from torch.optim import SGD, Optimizer

class MultiLayerModule (Module):
    layers: ModuleList
    layer_sizes: int

    def __init__(self, layer_sizes: List[int]):
        super().__init__()
        layer_sizes = [1] + layer_sizes
        self.layer_sizes = layer_sizes
        self.layers = ModuleList()
        for i in range(1, len(layer_sizes)):
            layer = Layer(layer_sizes[i-1], layer_sizes[i])
            if (i != len(layer_sizes)-1): layer.activation_function = LeakyReLU()
            self.layers.append(layer)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(-1)


class Layer (Module):
    weights: Tensor
    activation_function: Module

    def __init__(self, input_size: int, output_size: int, activation_function: Optional[Module] = None):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.rand(
            (input_size+1, output_size), dtype=torch.float, requires_grad=True))
        self.activation_function = activation_function
        if self.activation_function == None: self.activation_function = lambda x: x

    def forward(self, x: Tensor) -> Tensor:
        bias_size = x.size()
        bias_size = bias_size[:-1] + (1,)
        x = torch.cat((x, ones(bias_size, dtype=torch.float)), -1)
        # print(x)
        return self.activation_function(x @ self.weights)

def fit(model: Module, x: Tensor, y: Tensor, epochs: int = 2000, criterion: Optional[Module] = None, optimizer: Optional[Optimizer] = None,
        lr: float = 1e-6) -> np.ndarray:

    if (criterion is None):
        criterion = MSELoss(reduction='sum')
    if (optimizer is None):
        optimizer = SGD(model.parameters(), lr=lr)

    loss_history: np.ndarray = np.array([])

    for t in range(epochs):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        if t % 10 == 0:
            loss_history = np.append(loss_history, loss.detach().numpy())
            print(t, loss_history[-1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_history



x = torch.linspace(-math.pi, math.pi, 2000, dtype=torch.float)
# y = torch.sin(x)
y = -torch.ones(2000)

m: Module = MultiLayerModule([4,16,1])
plt.plot(fit(m,x,y, lr=1e-6, epochs=1000))
plt.show()
plt.plot(x, y, label="y")
plt.plot(x, m(x).detach(), label="pred_y")
plt.legend()
plt.show()