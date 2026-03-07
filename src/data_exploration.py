"""
Data Exploration Script for W&B Report (Section 2.1)

Logs a W&B table containing 5 sample images from each class
for MNIST or Fashion-MNIST.
"""

import argparse
import numpy as np
import wandb
from keras.datasets import mnist, fashion_mnist


def parse_arguments():
    parser = argparse.ArgumentParser(description="Data exploration for W&B report")

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"],
        help="Dataset to visualize",
    )

    parser.add_argument(
        "-w_p",
        "--wandb_project",
        type=str,
        required=True,
        help="Weights & Biases project name",
    )

    return parser.parse_args()


def load_raw_dataset(name):
    if name == "mnist":
        (X_train, y_train), _ = mnist.load_data()

        class_names = [str(i) for i in range(10)]

    elif name == "fashion_mnist":
        (X_train, y_train), _ = fashion_mnist.load_data()

        class_names = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]

    else:
        raise ValueError("Unsupported dataset")

    return X_train, y_train, class_names


def main():

    args = parse_arguments()

    run = wandb.init(
        project=args.wandb_project,
        name="data_exploration",
        job_type="analysis",
        config={"dataset": args.dataset},
    )

    X_train, y_train, class_names = load_raw_dataset(args.dataset)

    table = wandb.Table(columns=["class", "img1", "img2", "img3", "img4", "img5"])

    samples_per_class = 5

    for cls in range(10):
        indices = np.where(y_train == cls)[0][:samples_per_class]
    
        images = [wandb.Image(X_train[idx]) for idx in indices]
    
        table.add_data(
            class_names[cls],
            images[0],
            images[1],
            images[2],
            images[3],
            images[4]
        )

    wandb.log({"dataset_samples": table})

    wandb.finish()


if __name__ == "__main__":
    main()
