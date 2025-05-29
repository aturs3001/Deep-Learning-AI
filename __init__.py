# ============================================================
# __init__.py
# Author: Aric Hurkman
# Date: 2025-05-27
# Copyright (c) 2023 Aric Hurkman
# License: All rights reserved.
# Disclaimer: This code is for Portfolio and Educational purposes only.
# ============================================================
# Description:
# This module serves as the entry point for the Deep Learning Framework.
# It initializes the package and makes the necessary imports for the framework to function.
# The module also defines the public API for the framework, specifying which classes and functions
# are accessible to users when they import the package.
# Version and author information
__version__ = "1.0.0"
__author__ = "Aric Hurkman"
__email__ = "arichurkman@gmail.com"
# Import main classes and functions
from .neural_network import NeuralNetwork
from .layers import Layer, Dense, Activation, Dropout, BatchNormalization
from .activation import (
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
    
    # Optimizers
    'Optimizer', 'SGD', 'Adam', 'AdaGrad', 'RMSprop', 'AdamW',
    'OPTIMIZERS',
    
    # Loss functions
    'Loss', 
    'MeanSquaredError', 
    'MeanAbsoluteError',
    'BinaryCrossentropy',
    'CategoricalCrossentropy',
    'SparseCategoricalCrossentropy',
    'Huber',
    'LogCosh',
    'Hinge',
    'SquaredHinge',
    'KLDivergence',

    # Convenience functions
    'mse', 
    'mae',
    'binary_crossentropy',
    'categorical_crossentropy',

    # Dictionary
    'LOSS_FUNCTIONS'
]
# This module is part of a deep learning framework that provides a comprehensive set of tools
# for building and training neural networks. It is designed to be modular and extensible,
# allowing users to easily integrate new optimizers or modify existing ones.
# The framework supports a wide range of activation functions, optimizers, and loss functions,
# making it suitable for various deep learning tasks. It also includes utility functions
# for common operations such as computing loss and applying activation functions.
# The framework is intended for educational purposes and as a foundation for building
# more advanced deep learning models and applications.
# The examples provided in the package demonstrate how to use the framework effectively,
# showcasing its capabilities in a practical context. Users are encouraged to explore
# the provided examples and documentation to gain a deeper understanding of the framework
# and its features.
# The package is structured to facilitate easy access to its components,
# allowing users to import only the parts they need for their specific use cases.
# This modular design promotes code reuse and simplifies the process of building
# complex neural network architectures. The framework is built with performance and flexibility in mind,
# enabling users to experiment with different configurations and optimizations.
# The examples are designed to be clear and concise, making it easy for users to follow along.
# They illustrate the core concepts of the framework and provide a solid foundation
# for further exploration and experimentation. Users are encouraged to modify the examples
# to suit their needs and to experiment with different configurations and parameters.