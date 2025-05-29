# ============================================================
# Project: Deep Learning Framework
# Module: tests/test_activation.py
# Author: Aric Hurkman
# Date: 2025-05-27
# Copyright (c) 2023 Aric Hurkman
# License: All rights reserved.
# Disclaimer: This code is for Portfolio and Educational purposes only.
# ============================================================
# Description:
# Tests for activation functions.
# This module defines various activation functions and their derivatives,
# including sigmoid, tanh, ReLU, Leaky ReLU, ELU, Swish, Softmax, and Linear.
# It also includes test cases for these functions and their corresponding layer classes.
# Tests are implemented using pytest and numpy for numerical operations.
# Dependencies:
# - pytest: For running the tests.
# - numpy: For numerical operations and array manipulations.
# Dependencies are installed via pip:
# pip install pytest numpy
## Usage:
# To run the tests, execute the following command in the terminal:
# pytest tests/test_activation.py
## The tests will validate the functionality of the activation functions and their derivatives,
# ensuring they behave as expected across various inputs.
#============================================================

import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path to import deep_learning
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import activation functions from the deep_learning module
from deep_learning.activation import (
    sigmoid, sigmoid_prime, tanh, tanh_prime, relu, relu_prime,
    leaky_relu, leaky_relu_prime, elu, elu_prime, swish, swish_prime,
    softmax, softmax_prime, linear, linear_prime,
    Sigmoid, Tanh, ReLU, LeakyReLU, ELU, Swish, Softmax, Linear
)
# Test cases for activation functions and layers
class TestActivationFunctions:
    """Test cases for activation functions."""
    # Test cases for activation functions.
    def test_sigmoid(self):
        """Test sigmoid function."""
        x = np.array([0, 1, -1, 100, -100])
        result = sigmoid(x)
        
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        assert abs(result[0] - 0.5) < 1e-10
        assert result[3] > 0.99  # sigmoid(100) ≈ 1
        assert result[4] < 0.01  # sigmoid(-100) ≈ 0
    # end of test_sigmoid
    # end of class TestActivationFunctions

    # Test cases for activation function derivatives.
    def test_sigmoid_prime(self):
        """Test sigmoid derivative."""
        
        x = np.array([0, 1, -1])
        result = sigmoid_prime(x)
        
        assert np.all(result >= 0)
        assert abs(result[0] - 0.25) < 1e-10  # sigmoid'(0) = 0.25
    # end of test_sigmoid_prime
    # end of class TestActivationFunctions
    # Test cases for tanh activation function and its derivative.
    def test_tanh(self):
        """Test tanh function."""
        x = np.array([0, 1, -1])
        result = tanh(x)
        
        assert abs(result[0]) < 1e-10  # tanh(0) = 0
        assert result[1] > 0  # tanh(1) > 0
        assert result[2] < 0  # tanh(-1) < 0
        assert np.all(result >= -1)
        assert np.all(result <= 1)
    # end of test_tanh

    # Test cases for tanh derivative.
    def test_tanh_prime(self):
        """Test tanh derivative."""
        x = np.array([0, 1, -1])
        result = tanh_prime(x)
        
        assert abs(result[0] - 1.0) < 1e-10  # tanh'(0) = 1
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    # end of test_tanh_prime

    # Test cases for ReLU activation function and its derivative.
    def test_relu(self):
        """Test ReLU function."""
        x = np.array([-2, -1, 0, 1, 2])
        result = relu(x)
        
        expected = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_equal(result, expected)
    # end of test_relu
    
    # Test cases for ReLU derivative.
    def test_relu_prime(self):
        """Test ReLU derivative."""
        x = np.array([-2, -1, 0, 1, 2])
        result = relu_prime(x)
        
        expected = np.array([0, 0, 0, 1, 1])
        np.testing.assert_array_equal(result, expected)
    # end of test_relu_prime

    # Test cases for Leaky ReLU activation function and its derivative.
    def test_leaky_relu(self):
        """Test Leaky ReLU function."""
        x = np.array([-2, -1, 0, 1, 2])
        result = leaky_relu(x, alpha=0.1)
        
        expected = np.array([-0.2, -0.1, 0, 1, 2])
        np.testing.assert_array_almost_equal(result, expected)
    # end of test_leaky_relu

    # Test cases for Leaky ReLU derivative.
    def test_leaky_relu_prime(self):
        """Test Leaky ReLU derivative."""
        x = np.array([-2, -1, 0, 1, 2])
        result = leaky_relu_prime(x, alpha=0.1)
        
        expected = np.array([0.1, 0.1, 0.1, 1, 1])
        np.testing.assert_array_almost_equal(result, expected)
    # end of test_leaky_relu_prime
    # Test cases for ELU activation function and its derivative.
    def test_elu(self):
        """Test ELU function."""
        x = np.array([-2, -1, 0, 1, 2])
        result = elu(x, alpha=1.0)
        
        assert result[0] < 0  # ELU(-2) < 0
        assert result[1] < 0  # ELU(-1) < 0
        assert result[2] == 0  # ELU(0) = 0
        assert result[3] == 1  # ELU(1) = 1
        assert result[4] == 2  # ELU(2) = 2
    # end of test_elu
    # Test cases for ELU derivative.
    def test_elu_prime(self):
        """Test ELU derivative."""
        x = np.array([0, 1, 2])
        result = elu_prime(x, alpha=1.0)
        
        assert result[0] == 1  # ELU'(0) = 1
        assert result[1] == 1  # ELU'(1) = 1
        assert result[2] == 1  # ELU'(2) = 1
    # end of test_elu_prime

    # Test cases for Swish activation function and its derivative.
    def test_swish(self):
        """Test Swish function."""
        x = np.array([0, 1, -1])
        result = swish(x)
        
        assert abs(result[0]) < 1e-10  # swish(0) = 0
        assert result[1] > 0  # swish(1) > 0
        assert result[2] < 0  # swish(-1) < 0
    # end of test_swish

    # Test cases for Swish derivative.
    def test_swish_prime(self):
        """Test Swish derivative."""
        x = np.array([0, 1, -1])
        result = swish_prime(x)
        
        assert abs(result[0] - 0.5) < 1e-10  # swish'(0) = 0.5
    # end of test_swish_prime

    # Test cases for Softmax activation function and its derivative.
    def test_softmax(self):
        """Test Softmax function."""
        x = np.array([[1, 2, 3], [1, 1, 1]])
        result = softmax(x)
        
        # Each column should sum to 1
        column_sums = np.sum(result, axis=0)
        np.testing.assert_array_almost_equal(column_sums, [1, 1, 1])
        
        # All values should be positive
        assert np.all(result > 0)
    # end of test_softmax

    # Test cases for Softmax derivative.
    def test_softmax_prime(self):
        """Test Softmax derivative."""
        x = np.array([[1, 2], [1, 1]])
        result = softmax_prime(x)
        
        assert result.shape == x.shape
    # end of test_softmax_prime
    # Test cases for Linear activation function and its derivative.
    def test_linear(self):
        """Test Linear function."""
        x = np.array([1, 2, 3, -1, -2])
        result = linear(x)
        
        np.testing.assert_array_equal(result, x)
    # end of test_linear
    # Test cases for Linear derivative.
    def test_linear_prime(self):
        """Test Linear derivative."""
        x = np.array([1, 2, 3, -1, -2])
        result = linear_prime(x)
        
        expected = np.ones_like(x)
        np.testing.assert_array_equal(result, expected)
    # end of test_linear_prime
    # end of class TestActivationFunctions

# Test cases for activation layer classes.
class TestActivationLayers:
    """Test cases for activation layer classes."""
    
    def test_sigmoid_layer(self):
        """Test Sigmoid activation layer."""
        layer = Sigmoid()
        x = np.array([[0], [1], [-1]])
        
        output = layer.forward(x)
        assert output.shape == x.shape
        
        gradient = layer.backward(np.ones_like(output), 0.1)
        assert gradient.shape == x.shape
    
    def test_tanh_layer(self):
        """Test Tanh activation layer."""
        layer = Tanh()
        x = np.array([[0], [1], [-1]])
        
        output = layer.forward(x)
        assert output.shape == x.shape
        
        gradient = layer.backward(np.ones_like(output), 0.1)
        assert gradient.shape == x.shape
    
    def test_relu_layer(self):
        """Test ReLU activation layer."""
        layer = ReLU()
        x = np.array([[-1], [0], [1]])
        
        output = layer.forward(x)
        expected = np.array([[0], [0], [1]])
        np.testing.assert_array_equal(output, expected)
        
        gradient = layer.backward(np.ones_like(output), 0.1)
        expected_grad = np.array([[0], [0], [1]])
        np.testing.assert_array_equal(gradient, expected_grad)
    
    def test_leaky_relu_layer(self):
        """Test LeakyReLU activation layer."""
        layer = LeakyReLU(alpha=0.1)
        x = np.array([[-1], [0], [1]])
        
        output = layer.forward(x)
        expected = np.array([[-0.1], [0], [1]])
        np.testing.assert_array_almost_equal(output, expected)
    
    def test_elu_layer(self):
        """Test ELU activation layer."""
        layer = ELU(alpha=1.0)
        x = np.array([[0], [1], [2]])
        
        output = layer.forward(x)
        assert output.shape == x.shape
    
    def test_swish_layer(self):
        """Test Swish activation layer."""
        layer = Swish()
        x = np.array([[0], [1], [-1]])
        
        output = layer.forward(x)
        assert output.shape == x.shape
    
    def test_softmax_layer(self):
        """Test Softmax activation layer."""
        layer = Softmax()
        x = np.array([[1, 2], [2, 1], [3, 3]])
        
        output = layer.forward(x)
        assert output.shape == x.shape
        
        # Each column should sum to 1
        column_sums = np.sum(output, axis=0)
        np.testing.assert_array_almost_equal(column_sums, [1, 1])
    
    def test_linear_layer(self):
        """Test Linear activation layer."""
        layer = Linear()
        x = np.array([[1], [2], [3]])
        
        output = layer.forward(x)
        np.testing.assert_array_equal(output, x)
        
        gradient = layer.backward(np.ones_like(output), 0.1)
        np.testing.assert_array_equal(gradient, np.ones_like(x))

# end of class TestActivationLayers
# Run the tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__])