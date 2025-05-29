# ============================================================
# setup.py
# Author: Aric Hurkman
# Date: 2025-05-27
# Copyright (c) 2023 Aric Hurkman
# License: All rights reserved.
# Disclaimer: This code is for Portfolio and Educational purposes only.
# ============================================================
# Description:
# This module provides various optimization algorithms for training neural networks.
# It includes implementations of popular optimizers such as SGD, Adam, AdaGrad, RMSprop, and AdamW.
# These optimizers are designed to improve the convergence speed and performance of neural networks.
#     'leaky_relu', 'leaky_relu_prime', 'elu', 'elu_prime', 'swish', 'swish_prime',
#     'softmax', 'softmax_prime', 'linear', 'linear_prime',
#
#     # Activation layer classes
#     'Sigmoid', 'Tanh', 'ReLU', 'LeakyReLU', 'ELU', 'Swish', 'Softmax', 'Linear',
#
#     # Optimizers
#     'Optimizer', 'SGD', 'Adam', 'AdaGrad', 'RMSprop', 'AdamW',
#     'OPTIMIZERS',
#
#     # Loss functions
#     'Loss', 'MeanSquaredError', 'MeanAbsoluteError',
#     'BinaryCrossentropy', 'CategoricalCrossentropy', 'SparseCategoricalCrossentropy',
#     'Huber', 'LogCosh', 'Hinge', 'SquaredHinge', 'KLDivergence',
#     'mse', 'mae', 'binary_crossentropy', 'categorical_crossentropy',
#     'LOSS_FUNCTIONS'
# ]
# This module is part of a deep learning framework that provides a comprehensive set of tools
# for building and training neural networks. It is designed to be modular and extensible,
# allowing users to easily integrate new optimizers or modify existing ones.
# ============================================================
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deep-learning-framework",
    version="1.0.0",
    author="Aric Hurkman",
    author_email="arichurkman@gmail.com",
    description="A comprehensive deep learning framework built from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deep-learning-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.9",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dl-demo=examples.classification_demo:main",
        ],
    },
)