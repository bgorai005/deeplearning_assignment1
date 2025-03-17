# DA6401 Assignment 1 - Fashion-MNIST Classification Project
## Problem Statement
In this assignment, you will implement a feedforward neural network and manually code the backpropagation algorithm for training. You must use NumPy for all matrix and vector operations, without any automatic differentiation packages. The network will be trained on the Fashion-MNIST dataset to classify 28x28 grayscale images into 10 fashion categories. The project also explores various optimization techniques and hyperparameter tuning to enhance model performance. The implementation utilizes NumPy, Pandas, and Matplotlib, with Keras for dataset loading and Scikit-learn's train_test_split for data preprocessing.
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
## Hyperparameters

        Learning rate: 0.001, 0.0001

        Number of hidden layers: 3, 4, 5

        Number of nodes in each hidden layer: 32, 64, 128

        Activation function: sigmoid, relu, tanh

        Optimization algorithm:  SGD, Adam, RMSprop, mgd, Nadam, nestrov

        Batch size: 16, 32, 64

        Epochs: 5, 10

        Weight initialization: xavier, random
  After that find the best configuration on the basis of validation accuracy.

## Confusion Matrix
For the **best model**, a **confusion matrix** is plotted along with the corresponding **hyperparameters** to visualize classification performance.

## Predictions on MNIST Dataset
To further evaluate the model’s robustness, three **recommended hyperparameter sets** were selected and tested on the **MNIST dataset**. 


---
For more details, refer to `DL_Assignment1.ipynb` and the documented functions in the repository.
The wandb report link [Visit W&B](https://wandb.ai/)



