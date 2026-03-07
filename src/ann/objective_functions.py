"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np


class CrossEntropy:
    def forward(self, logits, y_true):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        self.probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(self.probs + 1e-12)) / m
        return loss

    def backward(self, y_true):
        return self.probs - y_true


class MSE:
    def forward(self, logits, y_true):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        self.probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        m = y_true.shape[0]
        loss = np.sum((self.probs - y_true) ** 2) / m
        return loss

    def backward(self, y_true):
        m = y_true.shape[0]
        return 2 * (self.probs - y_true) / m
