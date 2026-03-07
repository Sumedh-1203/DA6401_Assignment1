import numpy as np
from ann.neural_network import NeuralNetwork


class Dummy:
    dataset = "mnist"
    num_layers = 2
    hidden_size = [64, 64]
    activation = "relu"
    weight_init = "xavier"
    learning_rate = 0.001
    weight_decay = 0.0
    loss = "cross_entropy"
    optimizer = "sgd"


model = NeuralNetwork(Dummy())

# Forward check
X = np.random.randn(4, 784)
y = np.zeros((4, 10))
y[np.arange(4), np.random.randint(0, 10, 4)] = 1

logits = model.forward(X)
print("Logits shape:", logits.shape)

# Backward check
model.loss_fn.forward(logits, y)
model.backward(y, logits)

for i, layer in enumerate(model.layers):
    print(f"Layer {i} grad_W shape:", layer.grad_W.shape)
    print(f"Layer {i} grad_b shape:", layer.grad_b.shape)

# Weight update check
W_before = model.layers[0].W.copy()
model.update_weights()
print("Weights changed:", not np.allclose(W_before, model.layers[0].W))