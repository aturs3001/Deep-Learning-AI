# ============================================================
# activation.py
# Author: Aric Hurkman
# Date: 2025-05-27
# Copyright (c) 2023 Aric Hurkman
# License: All rights reserved.
# Disclaimer: This code is for Portfolio and Educational purposes only.
# ============================================================
# Description:
# activation.py for the Deep Learning Framework project
# This module defines various activation functions and their derivatives,
# as well as corresponding activation layer classes for use in deep learning models
import numpy as np
from .layers import Activation


# Activation functions
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    # Clip x to prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation function."""
    return np.tanh(x)


def tanh_prime(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh function."""
    return 1 - np.tanh(x) ** 2


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation function."""
    return np.maximum(0, x)


def relu_prime(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU function."""
    return (x > 0).astype(float)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU activation function."""
    return np.where(x > 0, x, alpha * x)


def leaky_relu_prime(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Derivative of Leaky ReLU function."""
    return np.where(x > 0, 1, alpha)


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Exponential Linear Unit activation function."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def elu_prime(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Derivative of ELU function."""
    return np.where(x > 0, 1, alpha * np.exp(x))


def swish(x: np.ndarray) -> np.ndarray:
    """Swish activation function."""
    return x * sigmoid(x)


def swish_prime(x: np.ndarray) -> np.ndarray:
    """Derivative of Swish function."""
    s = sigmoid(x)
    return s * (1 + x * (1 - s))


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation function."""
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def softmax_prime(x: np.ndarray) -> np.ndarray:
    """Derivative of softmax function."""
    s = softmax(x)
    return s * (1 - s)


def linear(x: np.ndarray) -> np.ndarray:
    """Linear activation function (identity)."""
    return x


def linear_prime(x: np.ndarray) -> np.ndarray:
    """Derivative of linear function."""
    return np.ones_like(x)


# Activation layer classes
class Sigmoid(Activation):
    """Sigmoid activation layer."""
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)


class Tanh(Activation):
    """Tanh activation layer."""
    def __init__(self):
        super().__init__(tanh, tanh_prime)


class ReLU(Activation):
    """ReLU activation layer."""
    def __init__(self):
        super().__init__(relu, relu_prime)


class LeakyReLU(Activation):
    """Leaky ReLU activation layer."""
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        super().__init__(
            lambda x: leaky_relu(x, alpha),
            lambda x: leaky_relu_prime(x, alpha)
        )


class ELU(Activation):
    """ELU activation layer."""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        super().__init__(
            lambda x: elu(x, alpha),
            lambda x: elu_prime(x, alpha)
        )


class Swish(Activation):
    """Swish activation layer."""
    def __init__(self):
        super().__init__(swish, swish_prime)


class Softmax(Activation):
    """Softmax activation layer."""
    def __init__(self):
        super().__init__(softmax, softmax_prime)


class Linear(Activation):
    """Linear activation layer."""
    def __init__(self):
        super().__init__(linear, linear_prime)


# Dictionary for easy access to activation functions
ACTIVATION_FUNCTIONS = {
    'sigmoid': (sigmoid, sigmoid_prime),
    'tanh': (tanh, tanh_prime),
    'relu': (relu, relu_prime),
    'leaky_relu': (leaky_relu, leaky_relu_prime),
    'elu': (elu, elu_prime),
    'swish': (swish, swish_prime),
    'softmax': (softmax, softmax_prime),
    'linear': (linear, linear_prime),
}


ACTIVATION_LAYERS = {
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'relu': ReLU,
    'leaky_relu': LeakyReLU,
    'elu': ELU,
    'swish': Swish,
    'softmax': Softmax,
    'linear': Linear,
}
