"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

import numpy as np


class BaseOptimizer:
    def __init__(self, layers, lr=0.001):
        self.layers = layers
        self.lr = lr

    def step(self):
        raise NotImplementedError


class SGD(BaseOptimizer):
    def step(self):
        for layer in self.layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b


class Momentum(BaseOptimizer):
    def __init__(self, layers, lr=0.001, beta=0.9):
        super().__init__(layers, lr)
        self.beta = beta
        self.v_W = [np.zeros_like(layer.W) for layer in layers]
        self.v_b = [np.zeros_like(layer.b) for layer in layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            self.v_W[i] = self.beta * self.v_W[i] + self.lr * layer.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + self.lr * layer.grad_b
            layer.W -= self.v_W[i]
            layer.b -= self.v_b[i]


class NAG(BaseOptimizer):
    def __init__(self, layers, lr=0.001, beta=0.9):
        super().__init__(layers, lr)
        self.beta = beta
        self.v_W = [np.zeros_like(layer.W) for layer in layers]
        self.v_b = [np.zeros_like(layer.b) for layer in layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            v_prev_W = self.v_W[i].copy()
            v_prev_b = self.v_b[i].copy()

            self.v_W[i] = self.beta * self.v_W[i] + self.lr * layer.grad_W
            self.v_b[i] = self.beta * self.v_b[i] + self.lr * layer.grad_b

            layer.W -= (-self.beta * v_prev_W + (1 + self.beta) * self.v_W[i])
            layer.b -= (-self.beta * v_prev_b + (1 + self.beta) * self.v_b[i])


class RMSProp(BaseOptimizer):
    def __init__(self, layers, lr=0.001, beta=0.9, eps=1e-8):
        super().__init__(layers, lr)
        self.beta = beta
        self.eps = eps
        self.s_W = [np.zeros_like(layer.W) for layer in layers]
        self.s_b = [np.zeros_like(layer.b) for layer in layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            self.s_W[i] = self.beta * self.s_W[i] + (1 - self.beta) * (layer.grad_W ** 2)
            self.s_b[i] = self.beta * self.s_b[i] + (1 - self.beta) * (layer.grad_b ** 2)

            layer.W -= self.lr * layer.grad_W / (np.sqrt(self.s_W[i]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.s_b[i]) + self.eps)


class Adam(BaseOptimizer):
    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(layers, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_W = [np.zeros_like(layer.W) for layer in layers]
        self.m_b = [np.zeros_like(layer.b) for layer in layers]
        self.v_W = [np.zeros_like(layer.W) for layer in layers]
        self.v_b = [np.zeros_like(layer.b) for layer in layers]
        self.t = 0

    def step(self):
        self.t += 1

        for i, layer in enumerate(self.layers):
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * layer.grad_W
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.grad_b

            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (layer.grad_W ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (layer.grad_b ** 2)

            m_hat_W = self.m_W[i] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_hat_W = self.v_W[i] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)

            layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.eps)
            layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)


class Nadam(BaseOptimizer):
    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(layers, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_W = [np.zeros_like(layer.W) for layer in layers]
        self.m_b = [np.zeros_like(layer.b) for layer in layers]
        self.v_W = [np.zeros_like(layer.W) for layer in layers]
        self.v_b = [np.zeros_like(layer.b) for layer in layers]
        self.t = 0

    def step(self):
        self.t += 1

        for i, layer in enumerate(self.layers):
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * layer.grad_W
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.grad_b

            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (layer.grad_W ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (layer.grad_b ** 2)

            m_hat_W = self.m_W[i] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_hat_W = self.v_W[i] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)

            nesterov_W = self.beta1 * m_hat_W + (1 - self.beta1) * layer.grad_W / (1 - self.beta1 ** self.t)
            nesterov_b = self.beta1 * m_hat_b + (1 - self.beta1) * layer.grad_b / (1 - self.beta1 ** self.t)

            layer.W -= self.lr * nesterov_W / (np.sqrt(v_hat_W) + self.eps)
            layer.b -= self.lr * nesterov_b / (np.sqrt(v_hat_b) + self.eps)


def get_optimizer(name, layers, lr=0.001):
    name = name.lower()

    if name == "sgd":
        return SGD(layers, lr)
    if name == "momentum":
        return Momentum(layers, lr)
    if name == "nag":
        return NAG(layers, lr)
    if name == "rmsprop":
        return RMSProp(layers, lr)
    if name == "adam":
        return Adam(layers, lr)
    if name == "nadam":
        return Nadam(layers, lr)

    raise ValueError("Unsupported optimizer")
