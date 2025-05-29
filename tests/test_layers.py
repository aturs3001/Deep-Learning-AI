# ============================================================
# test_layers.py
# Author: Aric Hurkman
# Date: 2025-05-27
# Copyright (c) 2023 Aric Hurkman
# License: All rights reserved.
# Disclaimer: This code is for Portfolio and Educational purposes only.
# ============================================================
# Description:
# Tests for neural network layers.
# This module contains unit tests for the Dense, Dropout, and BatchNormalization layers.
# Tests are implemented using pytest and numpy for numerical operations.
# ============================================================
# dependencies:
# pytest, numpy
# ============================================================
# usage:
# Run the tests using pytest:
# pytest tests/test_layers.py
# ============================================================

import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path to import deep_learning
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deep_learning.layers import Dense, Dropout, BatchNormalization
from deep_learning.activation import ReLU


class TestDense:
    """Test cases for Dense layer."""
    
    def test_dense_initialization(self):
        """Test Dense layer initialization."""
        layer = Dense(3, 2)
        assert layer.weights.shape == (2, 3)
        assert layer.bias.shape == (2, 1)
    
    def test_dense_forward(self):
        """Test Dense layer forward pass."""
        layer = Dense(3, 2)
        layer.weights = np.array([[1, 2, 3], [4, 5, 6]])
        layer.bias = np.array([[0.1], [0.2]])
        
        input_data = np.array([[1], [2], [3]])
        output = layer.forward(input_data)
        
        expected_output = np.array([[14.1], [32.2]])
        np.testing.assert_array_almost_equal(output, expected_output)
    
    def test_dense_backward(self):
        """Test Dense layer backward pass."""
        layer = Dense(2, 2)
        layer.weights = np.array([[1, 2], [3, 4]])
        layer.bias = np.array([[0.1], [0.2]])
        
        input_data = np.array([[1], [2]])
        layer.forward(input_data)
        
        output_gradient = np.array([[1], [1]])
        input_gradient = layer.backward(output_gradient, 0.1)
        
        assert input_gradient.shape == (2, 1)
    
    def test_dense_multiple_samples(self):
        """Test Dense layer with multiple samples."""
        layer = Dense(2, 3)
        input_data = np.array([[1, 2, 3], [4, 5, 6]])
        output = layer.forward(input_data)
        
        assert output.shape == (3, 3)


class TestDropout:
    """Test cases for Dropout layer."""
    
    def test_dropout_initialization(self):
        """Test Dropout layer initialization."""
        layer = Dropout(0.5)
        assert layer.rate == 0.5
        assert layer.training == True
    
    def test_dropout_forward_training(self):
        """Test Dropout forward pass in training mode."""
        layer = Dropout(0.5)
        input_data = np.ones((3, 10))
        
        # Set random seed for reproducible results
        np.random.seed(42)
        output = layer.forward(input_data)
        
        # Some values should be zeroed out
        assert not np.allclose(output, input_data)
        assert output.shape == input_data.shape
    
    def test_dropout_forward_inference(self):
        """Test Dropout forward pass in inference mode."""
        layer = Dropout(0.5)
        layer.set_training(False)
        input_data = np.ones((3, 10))
        
        output = layer.forward(input_data)
        
        # All values should be preserved in inference mode
        np.testing.assert_array_equal(output, input_data)
    
    def test_dropout_backward(self):
        """Test Dropout backward pass."""
        layer = Dropout(0.5)
        input_data = np.ones((3, 10))
        
        np.random.seed(42)
        layer.forward(input_data)
        
        output_gradient = np.ones((3, 10))
        input_gradient = layer.backward(output_gradient, 0.1)
        
        assert input_gradient.shape == output_gradient.shape


class TestBatchNormalization:
    """Test cases for Batch Normalization layer."""
    
    def test_batch_norm_initialization(self):
        """Test BatchNormalization initialization."""
        layer = BatchNormalization(3)
        assert layer.gamma.shape == (3, 1)
        assert layer.beta.shape == (3, 1)
        assert layer.running_mean.shape == (3, 1)
        assert layer.running_var.shape == (3, 1)
    
    def test_batch_norm_forward_training(self):
        """Test BatchNormalization forward pass in training mode."""
        layer = BatchNormalization(2)
        input_data = np.array([[1, 2, 3], [4, 5, 6]])
        
        output = layer.forward(input_data)
        
        assert output.shape == input_data.shape
        # Output should be normalized (approximately zero mean)
        assert abs(np.mean(output, axis=1)).max() < 1e-10
    
    def test_batch_norm_forward_inference(self):
        """Test BatchNormalization forward pass in inference mode."""
        layer = BatchNormalization(2)
        layer.set_training(False)
        input_data = np.array([[1, 2], [3, 4]])
        
        output = layer.forward(input_data)
        
        assert output.shape == input_data.shape
    
    def test_batch_norm_backward(self):
        """Test BatchNormalization backward pass."""
        layer = BatchNormalization(2)
        input_data = np.array([[1, 2, 3], [4, 5, 6]])
        
        layer.forward(input_data)
        output_gradient = np.ones((2, 3))
        input_gradient = layer.backward(output_gradient, 0.1)
        
        assert input_gradient.shape == input_data.shape


if __name__ == "__main__":
    pytest.main([__file__])