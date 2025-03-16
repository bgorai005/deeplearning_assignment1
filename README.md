# DA6401_assignment1
# Fashion-MNIST Classification Project

## Overview
This repository contains code for training and evaluating a neural network model for Fashion-MNIST image classification using various optimization techniques and hyperparameter tuning. The model is implemented in Python with libraries like NumPy, pandas ,Matplotlib only and Keras for dataset module for data loading, and Scikit-learn's `train_test_split` for data partitioning.

## Dataset
The Fashion-MNIST dataset consists of:
- **60,000 training images** and **10,000 testing images**
- **10 classes**, each representing a different type of fashion item
- Each image is a **28x28 grayscale** representation

Before training, the dataset undergoes preprocessing:
- **Flattening** each image into a 1D array
- **Normalization** of training, validation, and test data between 0 and 1
- **One-hot encoding** of labels into arrays of length 10 (one for each class)

## Project Goals
The goal of this project is to develop a deep learning model capable of accurately classifying Fashion-MNIST images into one of the 10 categories. This involves:
- Building a flexible neural network architecture with multiple hidden layers
- Using various activation functions and optimization algorithms
- Exploring hyperparameter tuning via  Bayesian Optimization technique.

## Optimization Algorithms
The following optimization techniques are implemented:
- Stochastic Gradient Descent (SGD)
- Momentum-based Gradient Descent
- Nesterov Accelerated Gradient
- RMSprop
- Adam
- Nadam

## Model Initialization
To initialize the neural network, the following parameters are required:
- **Number of hidden layers**
- **Number of nodes per hidden layer**
- **Weight initialization method**

The network initialization function is implemented in `DL_Assignment1.ipynb` as `intial_weights`.

## Forward Propagation
In the forward propagation step:
- Each hidden layer applies its respective activation function
- The output layer uses the softmax function for multi-class classification
- Predictions are generated for the training data

## Backward Propagation
Backward propagation involves:
- Performing forward propagation to obtain predictions
- Calculating loss(cross_entropy and squared error loss) between predictions and actual labels
- Computing gradients of the loss with respect to model parameters (weights and biases)
- Using optimization algorithms to update model parameters, minimizing the loss over a specified number of epochs

## Model Training
The `model_train` function trains the model with various hyperparameter configurations using Weights & Biases (W&B) sweep configurations. Training outputs include:
- Training loss
- Training accuracy
- Validation Accuracy
- Validation accuracy

## Additional Information
The neural network implementation and its optimizers are designed for efficient experimentation and learning, ensuring robustness and flexibility in hyperparameter tuning and model performance evaluation.


