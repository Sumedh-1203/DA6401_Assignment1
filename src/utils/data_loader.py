"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
from keras.datasets import mnist, fashion_mnist

def _one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


def _preprocess(X):
    X = X.astype(np.float32) / 255.0
    X = X.reshape(X.shape[0], -1)
    return X


def load_dataset(name):
    name = name.lower()

    if name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Unsupported dataset")

    X_train = _preprocess(X_train)
    X_test = _preprocess(X_test)

    y_train = _one_hot_encode(y_train)
    y_test = _one_hot_encode(y_test)

    return X_train, y_train, X_test, y_test
