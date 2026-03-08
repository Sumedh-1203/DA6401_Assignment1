"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import os
import argparse
import json
import numpy as np
from sklearn.metrics import f1_score
import wandb

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
    parser.add_argument("-o", "--optimizer", type=str, default="nadam")
    parser.add_argument("-nhl", "--num_layers", type=int, default=5)
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", default=[128,128,128,128,128])
    parser.add_argument("-a", "--activation", type=str, default="relu")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy")
    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier")
    parser.add_argument("-w_p", "--wandb_project", type=str, default="")
    parser.add_argument("-msp", "--model_save_path", type=str, default="src/best_model.npy")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)

    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()

    if args.wandb_project != "":
        run = wandb.init(project=args.wandb_project, config=vars(args))
        run_name = run.id
    else:
        run_name = "manual_run"

    run_dir = os.path.join("src", "experiments", run_name)

    os.makedirs(run_dir, exist_ok=True)

    X_train, y_train, X_test, y_test = load_dataset(args.dataset)

    if len(args.hidden_size) != args.num_layers:
        args.hidden_size = [args.hidden_size[0]] * args.num_layers

    model = NeuralNetwork(args)

    best_f1 = 0.0
    best_weights = None

    for epoch in range(args.epochs):
        model.train(X_train, y_train, epochs=1, batch_size=args.batch_size)

        # grads = model.layers[0].grad_W

        # wandb.log({
        #     "grad_n1": np.linalg.norm(grads[:,0]),
        #     "grad_n2": np.linalg.norm(grads[:,1]),
        #     "grad_n3": np.linalg.norm(grads[:,2]),
        #     "grad_n4": np.linalg.norm(grads[:,3]),
        #     "grad_n5": np.linalg.norm(grads[:,4])
        # })

        grad_norm = np.linalg.norm(model.layers[0].grad_W)

        logits = model.forward(X_test)
        activations = model.layers[0].A

        dead_ratio = np.mean(activations == 0)
        if args.wandb_project != "":
            wandb.log({
                "epoch": epoch + 1,
                "dead_neuron_ratio": dead_ratio
            })

        model.loss_fn.forward(logits, y_test)
        probs = model.loss_fn.probs

        preds = np.argmax(probs, axis=1)
        true = np.argmax(y_test, axis=1)

        f1 = f1_score(true, preds, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_weights = model.get_weights()

        print(f"Epoch {epoch+1}/{args.epochs} - F1: {f1:.4f}")

        if args.wandb_project != "":
            wandb.log({
                "epoch": epoch + 1,
                "f1_score": f1,
                "grad_norm_layer1": grad_norm
            })

    if best_weights is not None:
        np.save(os.path.join(run_dir, "model.npy"), best_weights) # type: ignore

        config = vars(args)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
    
    metrics = {
        "best_f1": best_f1
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("Training complete!")


if __name__ == '__main__':
    main()
