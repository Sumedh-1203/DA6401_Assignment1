# DA6401 Assignment 1  
## Multi-Layer Perceptron for Image Classification (NumPy)

Project by:
Sumedh Kulkarni
NA21B041

This repository contains an implementation of a **Multi-Layer Perceptron (MLP)** built entirely using **NumPy** for image classification on the **MNIST** and **Fashion-MNIST** datasets. The project includes implementations of forward propagation, backpropagation, different optimizers, activation functions, weight initialization strategies, and experiment tracking using **Weights & Biases (W&B)**.

---

## Repository Structure

da6401_assignment_1/
│
├── src/
│ ├── train.py # Training script
│ ├── inference.py # Model evaluation and error analysis
│ ├── data_exploration.py # Dataset visualization
│ │
│ ├── ann/
│ │ ├── neural_network.py
│ │ ├── neural_layer.py
│ │ ├── activations.py
│ │ ├── objective_functions.py
│ │ └── optimizers.py
│ │
│ └── utils/
│ └── data_loader.py
│
├── sweep.yaml # Hyperparameter sweep configuration
├── confusion_matrix.png
├── misclassified_examples.png
└── README.md

---

## Installation

Create a Python environment and install the required packages:
pip install numpy matplotlib scikit-learn wandb keras
Login to Weights & Biases:
wandb login

---

## Training the Model

Example command for training:

python src/train.py -d mnist -e 10 -b 32 -lr 0.001 -o nadam -nhl 5 -sz 128 128 128 128 128 -a relu -l cross_entropy -w_i xavier -w_p da6401_test


### Important Arguments

| Argument | Description |
|--------|-------------|
| `-d` | Dataset (`mnist` or `fashion_mnist`) |
| `-e` | Number of epochs |
| `-b` | Batch size |
| `-lr` | Learning rate |
| `-o` | Optimizer |
| `-nhl` | Number of hidden layers |
| `-sz` | Hidden layer sizes |
| `-a` | Activation function |
| `-l` | Loss function |
| `-w_i` | Weight initialization |
| `-w_p` | W&B project name |

---

## Hyperparameter Sweep

Run the sweep:
wandb sweep sweep.yaml
Start the sweep agent:
wandb agent <username>/da6401_test/<sweep_id>

The sweep explores hyperparameters such as:
- learning rate  
- optimizer  
- activation function  
- batch size  
- number of hidden layers  
- weight initialization  
- weight decay  

---

## Inference and Error Analysis

Evaluate the best model and generate visualizations:
python src/inference.py -m src/model.npy -nhl 5 -sz 128 128 128 128 128 -a relu
This script generates:
- `confusion_matrix.png`
- `misclassified_examples.png`

These plots are used for analyzing model performance and failure cases.

---

## Experiments Summary

### Best MNIST Configuration

- Optimizer: **Nadam**
- Activation: **ReLU**
- Hidden Layers: **5**
- Neurons per Layer: **128**
- Learning Rate: **0.001**
- Weight Initialization: **Xavier**

Best validation F1 score:
0.9792

---

## Fashion-MNIST Transfer Experiment

Based on the MNIST experiments, three configurations were tested on Fashion-MNIST:

| Architecture | Optimizer | Activation | F1 Score |
|-------------|-----------|------------|----------|
| 5 × 128 | Nadam | ReLU | **0.8773** |
| 5 × 128 | RMSProp | ReLU | 0.8603 |
| 5 × 128 | Nadam | Tanh | 0.8732 |

The configuration that performed best on MNIST (**Nadam + ReLU**) also achieved the highest accuracy on Fashion-MNIST. However, the performance is lower because Fashion-MNIST contains more visually similar classes such as shirts, coats, and pullovers, making the classification task more difficult.

---

## Weights & Biases Report

Full experiment analysis and visualizations:

https://wandb.ai/sumedh2-1203-indian-institute-of-technology-madras/da6401_test/reports/DA6401-Assignment-1--VmlldzoxNjEzNDIzNg


The report includes:

- dataset exploration  
- hyperparameter sweep analysis  
- optimizer comparison  
- vanishing gradient analysis  
- dead neuron investigation  
- error analysis  
- Fashion-MNIST transfer experiment  

---

## Key Observations

- Adaptive optimizers such as **Nadam** and **RMSProp** outperform standard SGD.  
- **ReLU activation** helps mitigate vanishing gradients compared to Sigmoid.  
- High learning rates can produce **dead neurons** in ReLU networks.  
- **Xavier initialization** is necessary to break symmetry between neurons.  
- Fashion-MNIST requires deeper architectures due to increased dataset complexity.

---

## Author

DA6401 – Introduction to Deep Learning  
Assignment 1 – Multi-Layer Perceptron Implementation using NumPy
