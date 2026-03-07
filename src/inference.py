"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset

# ADDED IMPORTS FOR ERROR ANALYSIS
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def parse_arguments():
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument("-m", "--model_path", type=str, default="best_model.npy")
    parser.add_argument("-d", "--dataset", type=str, default="mnist")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-nhl", "--num_layers", type=int, default=2)
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", default=[128, 128])
    parser.add_argument("-a", "--activation", type=str, default="relu")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy")
    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier")
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
    """
    args = parse_arguments()

    X_train, y_train, X_test, y_test = load_dataset(args.dataset)

    model = NeuralNetwork(args)

    weights = load_model(args.model_path)
    model.set_weights(weights)

    results = evaluate_model(model, X_test, y_test)

    print("Accuracy:", results["accuracy"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1:", results["f1"])

    print("Evaluation complete!")

    # ------------------------------
    # ERROR ANALYSIS SECTION
    # ------------------------------

    logits = results["logits"]
    probs = model.loss_fn.probs

    preds = np.argmax(probs, axis=1)
    true = np.argmax(y_test, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(true, preds)

    plt.figure(figsize=(8,6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Test Set")

    # Add numbers inside matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.savefig("confusion_matrix.png")
    plt.close()

    # Misclassified examples visualization
    mis_idx = np.where(preds != true)[0]

    fig, axes = plt.subplots(5,5, figsize=(8,8))

    for i, ax in enumerate(axes.flatten()):
        idx = mis_idx[i]

        ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"T:{true[idx]} P:{preds[idx]}")
        ax.axis("off")

    plt.suptitle("Misclassified Test Images")

    plt.savefig("misclassified_examples.png")
    plt.close()

    return results


if __name__ == '__main__':
    main()
    