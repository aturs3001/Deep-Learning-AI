# ============================================================
# optimizers.py
# Author: Aric Hurkman
# Date: 2025-05-27
# Copyright (c) 2023 Aric Hurkman
# License: All rights reserved.
# Disclaimer: This code is for Portfolio and Educational purposes only.
# ============================================================
# Description:
# This module defines various optimization algorithms for training neural networks.
# optimizers.py for the Deep Learning Framework project
# This module provides implementations of popular optimizers such as SGD, Adam, AdaGrad, RMSprop, and AdamW.
# ============================================================
import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """Abstract base class for optimizers."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
    
    @abstractmethod
    def update(self, layer, weights_gradient: np.ndarray, bias_gradient: np.ndarray):
        """Update layer parameters."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset optimizer state."""
        pass


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity_weights = {}
        self.velocity_bias = {}
    
    def update(self, layer, weights_gradient: np.ndarray, bias_gradient: np.ndarray):
        """Update parameters using SGD with momentum."""
        layer_id = id(layer)
        
        # Initialize velocity if not exists
        if layer_id not in self.velocity_weights:
            self.velocity_weights[layer_id] = np.zeros_like(weights_gradient)
            self.velocity_bias[layer_id] = np.zeros_like(bias_gradient)
        
        # Update velocity
        self.velocity_weights[layer_id] = (
            self.momentum * self.velocity_weights[layer_id] - 
            self.learning_rate * weights_gradient
        )
        self.velocity_bias[layer_id] = (
            self.momentum * self.velocity_bias[layer_id] - 
            self.learning_rate * bias_gradient
        )
        
        # Update parameters
        layer.weights += self.velocity_weights[layer_id]
        layer.bias += self.velocity_bias[layer_id]
    
    def reset(self):
        """Reset velocity."""
        self.velocity_weights.clear()
        self.velocity_bias.clear()


class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = {}  # First moment
        self.v_weights = {}  # Second moment
        self.m_bias = {}
        self.v_bias = {}
        self.t = 0  # Time step
    
    def update(self, layer, weights_gradient: np.ndarray, bias_gradient: np.ndarray):
        """Update parameters using Adam."""
        layer_id = id(layer)
        self.t += 1
        
        # Initialize moments if not exists
        if layer_id not in self.m_weights:
            self.m_weights[layer_id] = np.zeros_like(weights_gradient)
            self.v_weights[layer_id] = np.zeros_like(weights_gradient)
            self.m_bias[layer_id] = np.zeros_like(bias_gradient)
            self.v_bias[layer_id] = np.zeros_like(bias_gradient)
        
        # Update biased first moment estimate
        self.m_weights[layer_id] = (
            self.beta1 * self.m_weights[layer_id] + 
            (1 - self.beta1) * weights_gradient
        )
        self.m_bias[layer_id] = (
            self.beta1 * self.m_bias[layer_id] + 
            (1 - self.beta1) * bias_gradient
        )
        
        # Update biased second raw moment estimate
        self.v_weights[layer_id] = (
            self.beta2 * self.v_weights[layer_id] + 
            (1 - self.beta2) * (weights_gradient ** 2)
        )
        self.v_bias[layer_id] = (
            self.beta2 * self.v_bias[layer_id] + 
            (1 - self.beta2) * (bias_gradient ** 2)
        )
        
        # Compute bias-corrected first moment estimate
        m_weights_corrected = self.m_weights[layer_id] / (1 - self.beta1 ** self.t)
        m_bias_corrected = self.m_bias[layer_id] / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_weights_corrected = self.v_weights[layer_id] / (1 - self.beta2 ** self.t)
        v_bias_corrected = self.v_bias[layer_id] / (1 - self.beta2 ** self.t)
        
        # Update parameters
        layer.weights -= (
            self.learning_rate * m_weights_corrected / 
            (np.sqrt(v_weights_corrected) + self.epsilon)
        )
        layer.bias -= (
            self.learning_rate * m_bias_corrected / 
            (np.sqrt(v_bias_corrected) + self.epsilon)
        )
    
    def reset(self):
        """Reset moments and time step."""
        self.m_weights.clear()
        self.v_weights.clear()
        self.m_bias.clear()
        self.v_bias.clear()
        self.t = 0
    
    def reset(self):
        """Reset moments and time step."""
        self.m_weights.clear()
        self.v_weights.clear()
        self.m_bias.clear()
        self.v_bias.clear()
        self.t = 0


class AdaGrad(Optimizer):
    """AdaGrad optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.sum_squared_weights = {}
        self.sum_squared_bias = {}
    
    def update(self, layer, weights_gradient: np.ndarray, bias_gradient: np.ndarray):
        """Update parameters using AdaGrad."""
        layer_id = id(layer)
        
        # Initialize accumulated squared gradients if not exists
        if layer_id not in self.sum_squared_weights:
            self.sum_squared_weights[layer_id] = np.zeros_like(weights_gradient)
            self.sum_squared_bias[layer_id] = np.zeros_like(bias_gradient)
        
        # Accumulate squared gradients
        self.sum_squared_weights[layer_id] += weights_gradient ** 2
        self.sum_squared_bias[layer_id] += bias_gradient ** 2
        
        # Update parameters
        layer.weights -= (
            self.learning_rate * weights_gradient / 
            (np.sqrt(self.sum_squared_weights[layer_id]) + self.epsilon)
        )
        layer.bias -= (
            self.learning_rate * bias_gradient / 
            (np.sqrt(self.sum_squared_bias[layer_id]) + self.epsilon)
        )
    
    def reset(self):
        """Reset accumulated gradients."""
        self.sum_squared_weights.clear()
        self.sum_squared_bias.clear()


class RMSprop(Optimizer):
    """RMSprop optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.avg_squared_weights = {}
        self.avg_squared_bias = {}
    
    def update(self, layer, weights_gradient: np.ndarray, bias_gradient: np.ndarray):
        """Update parameters using RMSprop."""
        layer_id = id(layer)
        
        # Initialize moving average if not exists
        if layer_id not in self.avg_squared_weights:
            self.avg_squared_weights[layer_id] = np.zeros_like(weights_gradient)
            self.avg_squared_bias[layer_id] = np.zeros_like(bias_gradient)
        
        # Update moving average of squared gradients
        self.avg_squared_weights[layer_id] = (
            self.rho * self.avg_squared_weights[layer_id] + 
            (1 - self.rho) * (weights_gradient ** 2)
        )
        self.avg_squared_bias[layer_id] = (
            self.rho * self.avg_squared_bias[layer_id] + 
            (1 - self.rho) * (bias_gradient ** 2)
        )
        
        # Update parameters
        layer.weights -= (
            self.learning_rate * weights_gradient / 
            (np.sqrt(self.avg_squared_weights[layer_id]) + self.epsilon)
        )
        layer.bias -= (
            self.learning_rate * bias_gradient / 
            (np.sqrt(self.avg_squared_bias[layer_id]) + self.epsilon)
        )
    
    def reset(self):
        """Reset moving averages."""
        self.avg_squared_weights.clear()
        self.avg_squared_bias.clear()


class AdamW(Optimizer):
    """AdamW optimizer (Adam with weight decay)."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m_weights = {}
        self.v_weights = {}
        self.m_bias = {}
        self.v_bias = {}
        self.t = 0
    
    def update(self, layer, weights_gradient: np.ndarray, bias_gradient: np.ndarray):
        """Update parameters using AdamW."""
        layer_id = id(layer)
        self.t += 1
        
        # Initialize moments if not exists
        if layer_id not in self.m_weights:
            self.m_weights[layer_id] = np.zeros_like(weights_gradient)
            self.v_weights[layer_id] = np.zeros_like(weights_gradient)
            self.m_bias[layer_id] = np.zeros_like(bias_gradient)
            self.v_bias[layer_id] = np.zeros_like(bias_gradient)
        
        # Apply weight decay
        layer.weights *= (1 - self.learning_rate * self.weight_decay)
        
        # Update biased first moment estimate
        self.m_weights[layer_id] = (
            self.beta1 * self.m_weights[layer_id] + 
            (1 - self.beta1) * weights_gradient
        )
        self.m_bias[layer_id] = (
            self.beta1 * self.m_bias[layer_id] + 
            (1 - self.beta1) * bias_gradient
        )
        
        # Update biased second raw moment estimate
        self.v_weights[layer_id] = (
            self.beta2 * self.v_weights[layer_id] + 
            (1 - self.beta2) * (weights_gradient ** 2)
        )
        self.v_bias[layer_id] = (
            self.beta2 * self.v_bias[layer_id] + 
            (1 - self.beta2) * (bias_gradient ** 2)
        )
        
        # Compute bias-corrected first moment estimate
        m_weights_corrected = self.m_weights[layer_id] / (1 - self.beta1 ** self.t)
        m_bias_corrected = self.m_bias[layer_id] / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_weights_corrected = self.v_weights[layer_id] / (1 - self.beta2 ** self.t)
        v_bias_corrected = self.v_bias[layer_id] / (1 - self.beta2 ** self.t)
        
        # Update parameters
        layer.weights -= (
            self.learning_rate * m_weights_corrected / 
            (np.sqrt(v_weights_corrected) + self.epsilon)
        )
        layer.bias -= (
            self.learning_rate * m_bias_corrected / 
            (np.sqrt(v_bias_corrected) + self.epsilon)
        )
    
    def reset(self):
        """Reset moments and time step."""
        self.m_weights.clear()
        self.v_weights.clear()
        self.m_bias.clear()
        self.v_bias.clear()
        self.t = 0
        
# Dictionary for easy access to optimizers
OPTIMIZERS = {
    'sgd': SGD,
    'adam': Adam,
    'adagrad': AdaGrad,
    'rmsprop': RMSprop,
    'adamw': AdamW,
}