"""
Deep Learning Framework - A comprehensive neural network library built from scratch.

This package provides all the essential components for building and training
neural networks including layers, activation functions, optimizers, and loss functions.
"""

__version__ = "1.0.0"
__author__ = "Aric Hurkman"
__email__ = "arichurkman@gmail.com"

# Import main classes and functions
from .neural_network import NeuralNetwork
from .layers import Layer, Dense, Activation, Dropout, BatchNormalization
from .activation import (
    # Activation functions
    sigmoid, sigmoid_prime, tanh, tanh_prime, relu, relu_prime,
    leaky_relu, leaky_relu_prime, elu, elu_prime, swish, swish_prime,
    softmax, softmax_prime, linear, linear_prime,
    
    # Activation layer classes
    Sigmoid, Tanh, ReLU, LeakyReLU, ELU, Swish, Softmax, Linear,
    
    # Dictionaries
    ACTIVATION_FUNCTIONS, ACTIVATION_LAYERS
)
from .optimizers import (
    Optimizer, SGD, Adam, AdaGrad, RMSprop, AdamW,
    OPTIMIZERS
)
from .loss import (
    Loss, MeanSquaredError, MeanAbsoluteError, 
    BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy,
    Huber, LogCosh, Hinge, SquaredHinge, KLDivergence,
    
    # Convenience functions
    mse, mae, binary_crossentropy, categorical_crossentropy,
    
    # Dictionary
    LOSS_FUNCTIONS
)

# Define what gets imported with "from deep_learning import *"
__all__ = [
    # Main class
    'NeuralNetwork',
    
    # Layer classes
    'Layer', 'Dense', 'Activation', 'Dropout', 'BatchNormalization',
    
    # Activation functions
    'sigmoid', 'sigmoid_prime', 'tanh', 'tanh_prime', 'relu', 'relu_prime',
    'leaky_relu', 'leaky_relu_prime', 'elu', 'elu_prime', 'swish', 'swish_prime',
    'softmax', 'softmax_prime', 'linear', 'linear_prime',
    
    # Activation layer classes
    'Sigmoid', 'Tanh', 'ReLU', 'LeakyReLU', 'ELU', 'Swish', 'Softmax', 'Linear',
    
    # Optimizer classes
    'Optimizer', 'SGD', 'Adam', 'AdaGrad', 'RMSprop', 'AdamW',
    
    # Loss classes
    'Loss', 'MeanSquaredError', 'MeanAbsoluteError', 
    'BinaryCrossentropy', 'CategoricalCrossentropy', 'SparseCategoricalCrossentropy',
    'Huber', 'LogCosh', 'Hinge', 'SquaredHinge', 'KLDivergence',
    
    # Convenience functions
    'mse', 'mae', 'binary_crossentropy', 'categorical_crossentropy',
    
    # Dictionaries for easy access
    'ACTIVATION_FUNCTIONS', 'ACTIVATION_LAYERS', 'OPTIMIZERS', 'LOSS_FUNCTIONS'
]

# Package information
