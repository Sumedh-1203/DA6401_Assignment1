"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np


class ReLU:
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0.0, Z)

    def backward(self, dA):
        dZ = dA * (self.Z > 0)
        return dZ


class Sigmoid:
    def forward(self, Z):
        self.A = 1.0 / (1.0 + np.exp(-Z))
        return self.A

    def backward(self, dA):
        dZ = dA * self.A * (1.0 - self.A)
        return dZ


class Tanh:
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dA):
        dZ = dA * (1.0 - self.A ** 2)
        return dZ


class Softmax:
    def forward(self, Z):
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return A
