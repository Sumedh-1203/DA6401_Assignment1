"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument("-m", "--model_path", type=str, default="best_model.npy")
    parser.add_argument("-d", "--dataset", type=str, default="mnist")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-nhl", "--num_layers", type=int, default=2)
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", default=[128, 128])
    parser.add_argument("-a", "--activation", type=str, default="relu")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy")
    parser.add_argument("-wi", "--weight_init", type=str, default="xavier")
    parser.add_argument("-o", "--optimizer", type=str, default="sgd")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)

    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    return np.load(model_path, allow_pickle=True).item()


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    logits = model.forward(X_test)
    loss = model.loss_fn.forward(logits, y_test)

    probs = model.loss_fn.probs
    preds = np.argmax(probs, axis=1)
    true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(true, preds)
    precision = precision_score(true, preds, average="macro", zero_division=0)
    recall = recall_score(true, preds, average="macro", zero_division=0)
    f1 = f1_score(true, preds, average="macro", zero_division=0)

    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()

    X_train, y_train, X_test, y_test = load_dataset(args.dataset)

    if len(args.hidden_size) != args.num_layers:
        raise ValueError("hidden_size length must match num_layers")

    model = NeuralNetwork(args)

    weights = load_model(args.model_path)
    model.set_weights(weights)

    results = evaluate_model(model, X_test, y_test)

    print("Accuracy:", results["accuracy"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1:", results["f1"])

    print("Evaluation complete!")

    return results


if __name__ == '__main__':
    main()
