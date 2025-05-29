# ============================================================
# test_neural_network.py
# Author: Aric Hurkman
# Date: 2025-05-27
# Copyright (c) 2023 Aric Hurkman
# License: All rights reserved.
# Disclaimer: This code is for Portfolio and Educational purposes only.
# ============================================================
# Description:
# This module provides implementations of various optimization algorithms
# for training neural networks. It includes popular optimizers such as
# Stochastic Gradient Descent (SGD) and Adam.
# Test cases for the Neural Network class in the deep learning framework.
# This module tests the functionality of the NeuralNetwork class,
# including layer addition, forward pass, training, evaluation, and saving/loading the model.
# It uses pytest for testing and numpy for numerical operations.
# ============================================================
# Dependencies:
# - pytest: For running the tests.
# - numpy: For numerical operations and array manipulations.
# - sys, os, tempfile: For file handling and path management.
# ============================================================
# Usage:
# Run the tests using pytest:
# pytest tests/test_neural_network.py
# ============================================================


import pytest
import numpy as np
import sys
import os
import tempfile

# Add the parent directory to the path to import deep_learning
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deep_learning.neural_network import NeuralNetwork
from deep_learning.layers import Dense
from deep_learning.activation import ReLU, Sigmoid
from deep_learning.optimizers import SGD, Adam
from deep_learning.loss import MeanSquaredError, BinaryCrossentropy


class TestNeuralNetwork:
    """Test cases for Neural Network class."""
    
    def test_neural_network_initialization(self):
        """Test neural network initialization."""
        # Test with string parameters
        nn = NeuralNetwork(loss='mse', optimizer='adam')
        assert isinstance(nn.loss_fn, MeanSquaredError)
        assert isinstance(nn.optimizer, Adam)
        
        # Test with object parameters
        nn2 = NeuralNetwork(loss=BinaryCrossentropy(), optimizer=SGD())
        assert isinstance(nn2.loss_fn, BinaryCrossentropy)
        assert isinstance(nn2.optimizer, SGD)
    
    def test_add_layer(self):
        """Test adding layers to the network."""
        nn = NeuralNetwork(loss='mse')
        
        # Add a dense layer
        dense_layer = Dense(3, 2)
        nn.add(dense_layer)
        
        assert len(nn.layers) == 1
        assert nn.layers[0] == dense_layer
    
    def test_add_dense_with_activation(self):
        """Test adding dense layer with activation."""
        nn = NeuralNetwork(loss='mse')
        
        # Add dense layer with ReLU activation
        nn.add_dense(units=5, activation='relu', input_size=3)
        
        # Should have added 2 layers: Dense + ReLU
        assert len(nn.layers) == 2
        assert isinstance(nn.layers[0], Dense)
        assert isinstance(nn.layers[1], ReLU)
    
    def test_add_dense_auto_size(self):
        """Test adding dense layer with automatic input size detection."""
        nn = NeuralNetwork(loss='mse')
        
        # Add first layer with explicit input size
        nn.add_dense(units=5, activation='relu', input_size=3)
        
        # Add second layer without specifying input size
        nn.add_dense(units=2, activation='sigmoid')
        
        # Should have 4 layers: Dense + ReLU + Dense + Sigmoid
        assert len(nn.layers) == 4
        assert isinstance(nn.layers[2], Dense)
        assert isinstance(nn.layers[3], Sigmoid)
    
    def test_add_dropout(self):
        """Test adding dropout layer."""
        nn = NeuralNetwork(loss='mse')
        nn.add_dense(units=5, activation='relu', input_size=3)
        nn.add_dropout(rate=0.5)
        
        # Should have 3 layers: Dense + ReLU + Dropout
        assert len(nn.layers) == 3
        assert hasattr(nn.layers[2], 'rate')
        assert nn.layers[2].rate == 0.5
    
    def test_forward_pass(self):
        """Test forward pass through the network."""
        nn = NeuralNetwork(loss='mse')
        nn.add_dense(units=3, activation='relu', input_size=2)
        nn.add_dense(units=1, activation='sigmoid')
        
        # Test forward pass
        X = np.array([[1, 2], [3, 4]])
        output = nn.forward(X)
        
        assert output.shape == (1, 2)
        assert np.all(output >= 0)  # Sigmoid output should be positive
        assert np.all(output <= 1)  # Sigmoid output should be <= 1
    
    def test_predict(self):
        """Test prediction method."""
        nn = NeuralNetwork(loss='mse')
        nn.add_dense(units=3, activation='relu', input_size=2)
        nn.add_dense(units=1, activation='sigmoid')
        
        X = np.array([[1, 2], [3, 4]])
        predictions = nn.predict(X)
        
        assert predictions.shape == (1, 2)
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)
    
    def test_evaluate(self):
        """Test evaluation method."""
        nn = NeuralNetwork(loss='binary_crossentropy')
        nn.add_dense(units=3, activation='relu', input_size=2)
        nn.add_dense(units=1, activation='sigmoid')
        
        X = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 0]])
        
        loss, accuracy = nn.evaluate(X, y)
        
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss >= 0
        assert 0 <= accuracy <= 1
    
    def test_train_batch(self):
        """Test training on a single batch."""
        nn = NeuralNetwork(loss='mse', optimizer='sgd')
        nn.add_dense(units=3, activation='relu', input_size=2)
        nn.add_dense(units=1, activation='linear')
        
        X = np.array([[1, 2], [3, 4]])
        y = np.array([[0.5, 1.5]])
        
        # Get initial weights
        initial_weights = nn.layers[0].weights.copy()
        
        # Train one batch
        loss = nn.train_batch(X, y)
        
        # Weights should change
        assert not np.allclose(nn.layers[0].weights, initial_weights)
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_fit_simple(self):
        """Test fitting the model with simple data."""
        # Create simple regression problem
        np.random.seed(42)
        X = np.array([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]])
        y = np.array([[3, 6, 9, 12, 15]])  # y = 3 * x1
        
        nn = NeuralNetwork(loss='mse', optimizer='adam')
        nn.add_dense(units=5, activation='relu', input_size=2)
        nn.add_dense(units=1, activation='linear')
        
        # Train for a few epochs
        history = nn.fit(X, y, epochs=5, batch_size=2, verbose=False)
        
        # Check that training history is recorded
        assert len(history['loss']) == 5
        assert len(history['accuracy']) == 5
        assert all(isinstance(loss, float) for loss in history['loss'])
    
    def test_fit_with_validation(self):
        """Test fitting with validation data."""
        np.random.seed(42)
        X_train = np.array([[1, 2, 5, 7], [3, 4, 6, 8]])  # Shape: (2, 4) - features x samples
        y_train = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])  # Shape: (2, 4) - classes x samples
        X_val = np.array([[2, 4], [3, 5]])  # Shape: (2, 2)
        y_val = np.array([[0, 1], [1, 0]])  # Shape: (2, 2)
        
        nn = NeuralNetwork(loss='categorical_crossentropy', optimizer='adam')
        nn.add_dense(units=3, activation='relu', input_size=2)
        nn.add_dense(units=2, activation='softmax')
        
        history = nn.fit(X_train, y_train, epochs=3, 
                        validation_data=(X_val, y_val), verbose=False)
        
        # Check validation metrics are recorded
        assert len(history['val_loss']) == 3
        assert len(history['val_accuracy']) == 3
    
    def test_set_training_mode(self):
        """Test setting training mode."""
        nn = NeuralNetwork(loss='mse')
        nn.add_dense(units=3, activation='relu', input_size=2)
        nn.add_dropout(rate=0.5)
        
        # Test setting training mode
        nn.set_training(False)
        assert nn.layers[2].training == False
        
        nn.set_training(True)
        assert nn.layers[2].training == True
    
    def test_summary(self):
        """Test network summary (just check it runs without error)."""
        nn = NeuralNetwork(loss='mse', optimizer='adam')
        nn.add_dense(units=10, activation='relu', input_size=5)
        nn.add_dense(units=5, activation='relu')
        nn.add_dense(units=1, activation='sigmoid')
        
        # This should not raise an exception
        nn.summary()
    
    def test_save_and_load(self):
        """Test saving and loading the model."""
        # Create and train a simple model
        np.random.seed(42)
        nn = NeuralNetwork(loss='mse', optimizer='sgd')
        nn.add_dense(units=3, activation='relu', input_size=2)
        nn.add_dense(units=1, activation='linear')
        
        X = np.array([[1, 2], [3, 4]])
        y = np.array([[1.5, 3.5]])
        
        # Train a bit to create some history
        nn.fit(X, y, epochs=2, verbose=False)
        
        # Save original weights and history
        original_weights = [layer.weights.copy() for layer in nn.layers if hasattr(layer, 'weights')]
        original_history = nn.training_history.copy()
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            nn.save(temp_path)
            
            # Create new model and load
            nn2 = NeuralNetwork(loss='mse', optimizer='sgd')
            nn2.load(temp_path)
            
            # Check that weights and history are restored
            loaded_weights = [layer.weights.copy() for layer in nn2.layers if hasattr(layer, 'weights')]
            
            assert len(loaded_weights) == len(original_weights)
            for orig, loaded in zip(original_weights, loaded_weights):
                np.testing.assert_array_almost_equal(orig, loaded)
            
            assert nn2.training_history['loss'] == original_history['loss']
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_invalid_loss_function(self):
        """Test initialization with invalid loss function."""
        with pytest.raises(ValueError):
            NeuralNetwork(loss='invalid_loss')
    
    def test_invalid_optimizer(self):
        """Test initialization with invalid optimizer."""
        with pytest.raises(ValueError):
            NeuralNetwork(loss='mse', optimizer='invalid_optimizer')
    
    def test_invalid_activation(self):
        """Test adding layer with invalid activation."""
        nn = NeuralNetwork(loss='mse')
        
        with pytest.raises(ValueError):
            nn.add_dense(units=5, activation='invalid_activation', input_size=3)
    
    def test_missing_input_size(self):
        """Test adding first layer without input size."""
        nn = NeuralNetwork(loss='mse')
        
        with pytest.raises(ValueError):
            nn.add_dense(units=5, activation='relu')  # No input_size specified
    
    def test_classification_accuracy(self):
        """Test accuracy calculation for classification."""
        # Binary classification
        nn = NeuralNetwork(loss='binary_crossentropy')
        nn.add_dense(units=1, activation='sigmoid', input_size=2)
        
        X = np.array([[1, 2], [3, 4]])
        y = np.array([[1, 0]])
        
        loss, accuracy = nn.evaluate(X, y)
        assert 0 <= accuracy <= 1
        
        # Multi-class classification
        nn2 = NeuralNetwork(loss='categorical_crossentropy')
        nn2.add_dense(units=3, activation='softmax', input_size=2)
        
        y_multi = np.array([[1, 0], [0, 1], [0, 0]])
        loss2, accuracy2 = nn2.evaluate(X, y_multi)
        assert 0 <= accuracy2 <= 1


if __name__ == "__main__":
    pytest.main([__file__])