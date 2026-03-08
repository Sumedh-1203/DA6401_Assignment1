"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
from ann.neural_layer import NeuralLayer
from ann.optimizers import get_optimizer
from ann.objective_functions import CrossEntropy, MSE


class NeuralNetwork:
    def __init__(self, cli_args):
        self.args = cli_args

        self.num_hidden_layers = cli_args.num_layers
        self.hidden_sizes = cli_args.hidden_size
        self.activation = cli_args.activation
        self.weight_init = cli_args.weight_init
        self.learning_rate = cli_args.learning_rate
        self.weight_decay = cli_args.weight_decay
        self.loss_type = cli_args.loss
        self.optimizer_name = cli_args.optimizer

        self.layers = []
        self._build_network()

        if self.loss_type == "cross_entropy":
            self.loss_fn = CrossEntropy()
        elif self.loss_type == "mse":
            self.loss_fn = MSE()
        else:
            raise ValueError("Unsupported loss")

        self.optimizer = get_optimizer(
            self.optimizer_name,
            self.layers,
            lr=self.learning_rate
        )

    def _build_network(self):
        input_dim = 784
        output_dim = 10

        prev_dim = input_dim

        for i in range(self.num_hidden_layers):
            layer = NeuralLayer(
                prev_dim,
                self.hidden_sizes[i],
                activation=self.activation,
                weight_init=self.weight_init
            )
            self.layers.append(layer)
            prev_dim = self.hidden_sizes[i]

        self.layers.append(
            NeuralLayer(
                prev_dim,
                output_dim,
                activation=None,
                weight_init=self.weight_init
            )
        )

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, logits, y_true):
        grad_W_list = []
        grad_b_list = []

        m = y_true.shape[0]
        self.loss_fn.forward(logits, y_true)
        dZ = self.loss_fn.backward(y_true)

        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)

            if self.weight_decay > 0:
                layer.grad_W += (self.weight_decay / m) * layer.W

            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        grad_W_list.reverse()
        grad_b_list.reverse()
        
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)

        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.step()

    def train(self, X_train, y_train, epochs=1, batch_size=32):
        n = X_train.shape[0]

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_train = X_train[perm]
            y_train = y_train[perm]

            for i in range(0, n, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                logits = self.forward(X_batch)
                loss = self.loss_fn.forward(logits, y_batch)

                self.backward(logits, y_batch)
                self.update_weights()

    def evaluate(self, X, y):
        logits = self.forward(X)
        loss = self.loss_fn.forward(logits, y)

        probs = self.loss_fn.probs
        preds = np.argmax(probs, axis=1)
        true = np.argmax(y, axis=1)

        accuracy = np.mean(preds == true)

        return loss, accuracy

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
