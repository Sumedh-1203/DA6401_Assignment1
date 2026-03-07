"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from ann.activations import ReLU, Sigmoid, Tanh


class NeuralLayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str | None = None,
        weight_init: str = "random",
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation
        self.weight_init = weight_init

        self._initialize_weights()

        self.X = None
        self.Z = None
        self.A = None
        self.grad_W = None
        self.grad_b = None

        if activation is None:
            self.activation = None
        elif activation == "relu":
            self.activation = ReLU()
        elif activation == "sigmoid":
            self.activation = Sigmoid()
        elif activation == "tanh":
            self.activation = Tanh()
        else:
            raise ValueError("Unsupported activation")

    def _initialize_weights(self):
        if self.weight_init == "random":
            self.W = np.random.randn(self.in_features, self.out_features) * 0.01
        elif self.weight_init == "xavier":
            limit = np.sqrt(2.0 / (self.in_features + self.out_features))
            self.W = np.random.randn(self.in_features, self.out_features) * limit
        elif self.weight_init == "zeros":
            self.W = np.zeros((self.in_features, self.out_features))
            self.b = np.zeros((1, self.out_features))
        else:
            raise ValueError("Unsupported weight initialization")

        self.b = np.zeros((1, self.out_features))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        self.Z = np.dot(X, self.W) + self.b

        if self.activation is not None:
            self.A = self.activation.forward(self.Z)
        else:
            self.A = self.Z  # logits

        return self.A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        if self.X is None:
            raise RuntimeError("backward() called before forward()")
        
        if self.activation is not None:
            dZ = self.activation.backward(dA)
        else:
            dZ = dA

        m = self.X.shape[0]

        self.grad_W = np.dot(self.X.T, dZ) / m
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) / m

        dX = np.dot(dZ, self.W.T)

        return dX
