"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import json
import numpy as np
from sklearn.metrics import f1_score

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument("-d", "--dataset", type=str, default="mnist")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-o", "--optimizer", type=str, default="sgd")
    parser.add_argument("-nhl", "--num_layers", type=int, default=2)
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", default=[128,128])
    parser.add_argument("-a", "--activation", type=str, default="relu")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy")
    parser.add_argument("-wi", "--weight_init", type=str, default="xavier")
    parser.add_argument("-wp", "--wandb_project", type=str, default="")
    parser.add_argument("-msp", "--model_save_path", type=str, default="src/best_model.npy")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)

    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()

    X_train, y_train, X_test, y_test = load_dataset(args.dataset)

    if len(args.hidden_size) != args.num_layers:
        raise ValueError("Number of hidden layer sizes must match num_layers")

    model = NeuralNetwork(args)

    best_f1 = 0.0
    best_weights = None

    for epoch in range(args.epochs):
        model.train(X_train, y_train, epochs=1, batch_size=args.batch_size)

        logits = model.forward(X_test)
        model.loss_fn.forward(logits, y_test)
        probs = model.loss_fn.probs

        preds = np.argmax(probs, axis=1)
        true = np.argmax(y_test, axis=1)

        f1 = f1_score(true, preds, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_weights = model.get_weights()

        print(f"Epoch {epoch+1}/{args.epochs} - F1: {f1:.4f}")

    if best_weights is not None:
        np.save("src/best_model.npy", best_weights) # type: ignore

        config = vars(args)
        with open("src/best_config.json", "w") as f:
            json.dump(config, f, indent=4)

    print("Training complete!")


if __name__ == '__main__':
    main()
