"""
Neural network layers implementation.
"""

import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class Layer(ABC):
    """Abstract base class for all layers."""
    
    def __init__(self):
        self.input = None
        self.output = None
    
    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through the layer."""
        pass
    
    @abstractmethod
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass through the layer."""
        pass


class Dense(Layer):
    """Fully connected (dense) layer."""
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size).astype(np.float64) * 0.1
        self.bias = np.zeros((output_size, 1), dtype=np.float64)
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through dense layer."""
        self.input = input_data
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass through dense layer."""
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        # Update parameters - ensure proper broadcasting
        self.weights = self.weights - learning_rate * weights_gradient
        self.bias = self.bias - learning_rate * output_gradient
        
        return input_gradient


class Activation(Layer):
    """Base activation layer."""
    
    def __init__(self, activation_func, activation_prime):
        super().__init__()
        self.activation = activation_func
        self.activation_prime = activation_prime
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through activation layer."""
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass through activation layer."""
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Dropout(Layer):
    """Dropout layer for regularization."""
    
    def __init__(self, rate: float = 0.5):
        super().__init__()
        self.rate = rate
        self.mask = None
        self.training = True
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through dropout layer."""
        self.input = input_data
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=input_data.shape) / (1 - self.rate)
            self.output = input_data * self.mask
        else:
            self.output = input_data
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass through dropout layer."""
        if self.training:
            return output_gradient * self.mask
        else:
            return output_gradient
    
    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training


class BatchNormalization(Layer):
    """Batch normalization layer."""
    
    def __init__(self, input_size: int, momentum: float = 0.9, epsilon: float = 1e-5):
        super().__init__()
        self.gamma = np.ones((input_size, 1), dtype=np.float64)
        self.beta = np.zeros((input_size, 1), dtype=np.float64)
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = np.zeros((input_size, 1), dtype=np.float64)
        self.running_var = np.ones((input_size, 1), dtype=np.float64)
        self.training = True
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through batch normalization layer."""
        self.input = input_data
        
        if self.training:
            # Calculate batch statistics
            self.mean = np.mean(input_data, axis=1, keepdims=True)
            self.var = np.var(input_data, axis=1, keepdims=True)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
            
            # Normalize
            self.normalized = (input_data - self.mean) / np.sqrt(self.var + self.epsilon)
        else:
            # Use running statistics during inference
            self.normalized = (input_data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        self.output = self.gamma * self.normalized + self.beta
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Backward pass through batch normalization layer."""
        m = output_gradient.shape[1]
        
        # Gradients for gamma and beta
        gamma_gradient = np.sum(output_gradient * self.normalized, axis=1, keepdims=True)
        beta_gradient = np.sum(output_gradient, axis=1, keepdims=True)
        
        # Gradient for normalized input
        normalized_gradient = output_gradient * self.gamma
        
        # Gradient for input
        var_gradient = np.sum(normalized_gradient * (self.input - self.mean), axis=1, keepdims=True)
        var_gradient *= -0.5 * (self.var + self.epsilon) ** (-1.5)
        
        mean_gradient = np.sum(normalized_gradient * -1 / np.sqrt(self.var + self.epsilon), axis=1, keepdims=True)
        mean_gradient += var_gradient * np.sum(-2 * (self.input - self.mean), axis=1, keepdims=True) / m
        
        input_gradient = normalized_gradient / np.sqrt(self.var + self.epsilon)
        input_gradient += var_gradient * 2 * (self.input - self.mean) / m
        input_gradient += mean_gradient / m
        
        # Update parameters
        self.gamma -= learning_rate * gamma_gradient
        self.beta -= learning_rate * beta_gradient
        
        return input_gradient
    
    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training