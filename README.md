# DA6401 Assignment 1 - Fashion-MNIST Classification Project

## Overview
This repository contains code for training and evaluating a neural network model for Fashion-MNIST image classification. The project explores various optimization techniques and hyperparameter tuning to enhance model performance. The implementation is done using **NumPy, Pandas, Matplotlib, and Keras (for dataset loading)**, while **Scikit-learn's `train_test_split`** is used for data partitioning.

## Dataset
The **Fashion-MNIST** dataset comprises:
- **60,000 training images** and **10,000 testing images**
- **10 classes**, representing different fashion items
- Each image is a **28x28 grayscale representation**

### **Preprocessing Steps**
Before training, the dataset undergoes the following preprocessing:
- **Flattening**: Each image is reshaped into a 1D array
- **Normalization**: Training, validation, and test data values are scaled between **0 and 1**
- **One-hot encoding**: Labels are converted into an array of length **10** (one for each class)

## Project Goals
The objective of this project is to build a **deep learning model** capable of accurately classifying Fashion-MNIST images into one of the **10 categories**. This involves:
- Designing a **flexible neural network architecture** with multiple hidden layers
- Implementing various **activation functions and optimization algorithms**
- **Fine-tuning hyperparameters** using **Bayesian Optimization**

## Optimization Algorithms
The project implements the following optimization techniques:
- **Stochastic Gradient Descent (SGD)**
- **Momentum-based Gradient Descent**
- **Nesterov Accelerated Gradient (NAG)**
- **RMSprop**
- **Adam**
- **Nadam**

## Model Initialization
To initialize the neural network, the following parameters are defined:
- **Number of hidden layers**
- **Number of nodes per hidden layer**
- **Weight initialization method**

The initialization function is implemented in `DL_Assignment1.ipynb` as `initialize_weights`.

## Forward Propagation
The forward propagation step involves:
- Applying the **activation function** at each hidden layer
- Using the **softmax function** at the output layer for multi-class classification
- Generating predictions for the training data

## Backward Propagation
Backward propagation is performed as follows:
- **Forward propagation** is executed to compute predictions
- The **loss function** (Cross-Entropy or Mean Squared Error) is calculated
- **Gradients** of the loss are computed with respect to weights and biases
- **Optimization algorithms** update model parameters to minimize the loss over multiple epochs

## Model Training
The `model_train` function executes the training process with different hyperparameter configurations using **Weights & Biases (W&B) sweep configurations**. The training results include:
- **Training loss**
- **Training accuracy**
- **Validation loss**
- **Validation accuracy**

## Confusion Matrix
For the **best model**, a **confusion matrix** is plotted along with the corresponding **hyperparameters** to visualize classification performance.

## Predictions on MNIST Dataset
To further evaluate the model’s robustness, three **recommended hyperparameter sets** were selected and tested on the **MNIST dataset**. The configurations used are:
- **Configuration 1**
- **Configuration 2**
- **Configuration 3**


---
For more details, refer to `DL_Assignment1.ipynb` and the documented functions in the repository.

