# ============================================================
# examples/__init__.py
# Author: Aric Hurkman
# Date: 2025-05-27
# Copyright (c) 2023 Aric Hurkman
# License: All rights reserved.
# Disclaimer: This code is for Portfolio and Educational purposes only.
# ============================================================
# Description:
# This package contains example scripts demonstrating the usage of the
# Deep Learning Framework. These examples cover a range of topics,
# including model training, evaluation, and inference.
# The examples are designed to be educational and showcase the capabilities
# of the framework in a practical context.
# Version and author information
__version__ = "1.0.0"
__author__ = "Aric Hurkman"
__email__ = "aric.hurkman@example.com"
# Import example scripts
from .mnist_example import mnist_example
from .cifar10_example import cifar10_example
from .custom_dataset_example import custom_dataset_example
from .transfer_learning_example import transfer_learning_example
from .hyperparameter_tuning_example import hyperparameter_tuning_example
from .visualization_example import visualization_example
from .model_saving_example import model_saving_example
from .model_loading_example import model_loading_example
from .data_augmentation_example import data_augmentation_example
from .model_evaluation_example import model_evaluation_example
# Define what gets imported with "from examples import *"
__all__ = [
    'mnist_example',
    'cifar10_example',
    'custom_dataset_example',
    'transfer_learning_example',
    'hyperparameter_tuning_example',
    'visualization_example',
    'model_saving_example',
    'model_loading_example',
    'data_augmentation_example',
    'model_evaluation_example'
]
# This allows users to import all example scripts easily
# when using "from examples import *".
# This is useful for educational purposes and quick access to example code.
# The examples are intended to be run independently and demonstrate
# various features of the Deep Learning Framework.
# Each example script is self-contained and can be executed to see
# the framework in action. They serve as a practical guide
# for users to understand how to use the framework effectively.
# The examples are designed to be clear and concise,
# making it easy for users to follow along and learn
# the key concepts of deep learning and neural networks.
# The examples also provide a foundation for users to build upon
# as they develop their own deep learning applications.
# The examples are structured to cover a variety of use cases,
# including image classification, custom dataset handling,
# transfer learning, hyperparameter tuning, and more.
# Users are encouraged to explore these examples and modify them
# to suit their own needs, enhancing their understanding of the framework
# and deep learning concepts.
# The examples are a valuable resource for both beginners and advanced users,
# providing practical insights into the capabilities of the Deep Learning Framework.
# The examples are also a great starting point for users
# who want to experiment with different neural network architectures,
# optimization techniques, and data preprocessing methods.
# By running these examples, users can gain hands-on experience
# with the framework and develop their skills in deep learning.
# The examples are maintained and updated regularly to ensure compatibility
# with the latest version of the framework and to incorporate user feedback.