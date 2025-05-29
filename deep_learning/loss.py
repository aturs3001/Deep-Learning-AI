# ============================================================
# loss.py
# Author: Aric Hurkman
# Date: 2025-05-27
# Copyright (c) 2023 Aric Hurkman
# License: All rights reserved.
# Disclaimer: This code is for Portfolio and Educational purposes only.
# ============================================================
# Description:
# loss.py for the Deep Learning Framework project
# This module defines various loss functions and their gradients,
# which are used to evaluate the performance of models during training.
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class Loss(ABC):
    """Abstract base class for loss functions."""
    
    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate the loss."""
        pass
    
    @abstractmethod
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the loss."""
        pass


class MeanSquaredError(Loss):
    """Mean Squared Error loss function."""
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate MSE loss."""
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Calculate MSE gradient."""
        return 2 * (y_pred - y_true) / y_pred.shape[1]


class MeanAbsoluteError(Loss):
    """Mean Absolute Error loss function."""
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate MAE loss."""
        return np.mean(np.abs(y_pred - y_true))
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Calculate MAE gradient."""
        return np.sign(y_pred - y_true) / y_pred.shape[1]


class BinaryCrossentropy(Loss):
    """Binary Cross-entropy loss function."""
    
    def __init__(self, epsilon: float = 1e-15):
        self.epsilon = epsilon
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate binary cross-entropy loss."""
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Calculate binary cross-entropy gradient."""
        # Clip predictions to prevent division by zero
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_pred.shape[1]


class CategoricalCrossentropy(Loss):
    """Categorical Cross-entropy loss function."""
    
    def __init__(self, epsilon: float = 1e-15):
        self.epsilon = epsilon
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate categorical cross-entropy loss."""
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Calculate categorical cross-entropy gradient."""
        # Clip predictions to prevent division by zero
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -y_true / y_pred / y_pred.shape[1]


class SparseCategoricalCrossentropy(Loss):
    """Sparse Categorical Cross-entropy loss function."""
    
    def __init__(self, epsilon: float = 1e-15):
        self.epsilon = epsilon
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate sparse categorical cross-entropy loss."""
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Convert sparse labels to probabilities
        batch_size = y_pred.shape[1]
        loss = 0
        for i in range(batch_size):
            true_class = int(y_true[0, i])
            loss += -np.log(y_pred[true_class, i])
        
        return loss / batch_size
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Calculate sparse categorical cross-entropy gradient."""
        # Clip predictions to prevent division by zero
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Initialize gradient
        gradient = np.zeros_like(y_pred)
        batch_size = y_pred.shape[1]
        
        for i in range(batch_size):
            true_class = int(y_true[0, i])
            gradient[true_class, i] = -1 / y_pred[true_class, i]
        
        return gradient / batch_size


class Huber(Loss):
    """Huber loss function (smooth combination of MSE and MAE)."""
    
    def __init__(self, delta: float = 1.0):
        self.delta = delta
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate Huber loss."""
        error = y_pred - y_true
        abs_error = np.abs(error)
        
        quadratic = np.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        
        return np.mean(0.5 * quadratic**2 + self.delta * linear)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Calculate Huber gradient."""
        error = y_pred - y_true
        abs_error = np.abs(error)
        
        gradient = np.where(
            abs_error <= self.delta,
            error,
            self.delta * np.sign(error)
        )
        
        return gradient / y_pred.shape[1]


class LogCosh(Loss):
    """Log-Cosh loss function."""
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate Log-Cosh loss."""
        error = y_pred - y_true
        return np.mean(np.log(np.cosh(error)))
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Calculate Log-Cosh gradient."""
        error = y_pred - y_true
        return np.tanh(error) / y_pred.shape[1]


class Hinge(Loss):
    """Hinge loss function for SVM-style classification."""
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate Hinge loss."""
        return np.mean(np.maximum(0, 1 - y_true * y_pred))
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Calculate Hinge gradient."""
        gradient = np.where(y_true * y_pred < 1, -y_true, 0)
        return gradient / y_pred.shape[1]


class SquaredHinge(Loss):
    """Squared Hinge loss function."""
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate Squared Hinge loss."""
        hinge = np.maximum(0, 1 - y_true * y_pred)
        return np.mean(hinge ** 2)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Calculate Squared Hinge gradient."""
        hinge = np.maximum(0, 1 - y_true * y_pred)
        gradient = np.where(y_true * y_pred < 1, -2 * y_true * hinge, 0)
        return gradient / y_pred.shape[1]


class KLDivergence(Loss):
    """Kullback-Leibler Divergence loss function."""
    
    def __init__(self, epsilon: float = 1e-15):
        self.epsilon = epsilon
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate KL Divergence loss."""
        # Clip to prevent log(0)
        y_pred = np.clip(y_pred, self.epsilon, 1)
        y_true = np.clip(y_true, self.epsilon, 1)
        
        return np.mean(np.sum(y_true * np.log(y_true / y_pred), axis=0))
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Calculate KL Divergence gradient."""
        # Clip to prevent division by zero
        y_pred = np.clip(y_pred, self.epsilon, 1)
        return -y_true / y_pred / y_pred.shape[1]


# Convenience functions
def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate Mean Squared Error."""
    return np.mean((y_pred - y_true) ** 2)


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_pred - y_true))


def binary_crossentropy(y_pred: np.ndarray, y_true: np.ndarray, epsilon: float = 1e-15) -> float:
    """Calculate Binary Cross-entropy."""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def categorical_crossentropy(y_pred: np.ndarray, y_true: np.ndarray, epsilon: float = 1e-15) -> float:
    """Calculate Categorical Cross-entropy."""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))


# Dictionary for easy access to loss functions
LOSS_FUNCTIONS = {
    'mse': MeanSquaredError,
    'mae': MeanAbsoluteError,
    'binary_crossentropy': BinaryCrossentropy,
    'categorical_crossentropy': CategoricalCrossentropy,
    'sparse_categorical_crossentropy': SparseCategoricalCrossentropy,
    'huber': Huber,
    'log_cosh': LogCosh,
    'hinge': Hinge,
    'squared_hinge': SquaredHinge,
    'kl_divergence': KLDivergence,
}