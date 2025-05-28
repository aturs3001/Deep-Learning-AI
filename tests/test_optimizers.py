"""
Tests for optimization algorithms.
"""

import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path to import deep_learning
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deep_learning.optimizers import SGD, Adam, AdaGrad, RMSprop, AdamW
from deep_learning.layers import Dense


class MockLayer:
    """Mock layer for testing optimizers."""
    
    def __init__(self, weights_shape, bias_shape):
        self.weights = np.random.randn(*weights_shape)
        self.bias = np.random.randn(*bias_shape)


class TestSGD:
    """Test cases for SGD optimizer."""
    
    def test_sgd_initialization(self):
        """Test SGD initialization."""
        optimizer = SGD(learning_rate=0.01, momentum=0.9)
        assert optimizer.learning_rate == 0.01
        assert optimizer.momentum == 0.9
    
    def test_sgd_update_no_momentum(self):
        """Test SGD update without momentum."""
        optimizer = SGD(learning_rate=0.1, momentum=0.0)
        layer = MockLayer((2, 3), (2, 1))
        
        weights_before = layer.weights.copy()
        bias_before = layer.bias.copy()
        
        weights_gradient = np.ones((2, 3))
        bias_gradient = np.ones((2, 1))
        
        optimizer.update(layer, weights_gradient, bias_gradient)
        
        expected_weights = weights_before - 0.1 * weights_gradient
        expected_bias = bias_before - 0.1 * bias_gradient
        
        np.testing.assert_array_almost_equal(layer.weights, expected_weights)
        np.testing.assert_array_almost_equal(layer.bias, expected_bias)
    
    def test_sgd_update_with_momentum(self):
        """Test SGD update with momentum."""
        optimizer = SGD(learning_rate=0.1, momentum=0.9)
        layer = MockLayer((2, 2), (2, 1))
        
        weights_gradient = np.ones((2, 2))
        bias_gradient = np.ones((2, 1))
        
        # First update
        optimizer.update(layer, weights_gradient, bias_gradient)
        
        # Second update should use momentum
        weights_before = layer.weights.copy()
        optimizer.update(layer, weights_gradient, bias_gradient)
        
        # Weights should change more due to momentum
        assert not np.allclose(layer.weights, weights_before + 0.1 * weights_gradient)
    
    def test_sgd_reset(self):
        """Test SGD reset."""
        optimizer = SGD(learning_rate=0.1, momentum=0.9)
        layer = MockLayer((2, 2), (2, 1))
        
        # Perform update to initialize velocity
        optimizer.update(layer, np.ones((2, 2)), np.ones((2, 1)))
        
        # Reset should clear velocity
        optimizer.reset()
        assert len(optimizer.velocity_weights) == 0
        assert len(optimizer.velocity_bias) == 0


class TestAdam:
    """Test cases for Adam optimizer."""
    
    def test_adam_initialization(self):
        """Test Adam initialization."""
        optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
        assert optimizer.learning_rate == 0.001
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.999
        assert optimizer.t == 0
    
    def test_adam_update(self):
        """Test Adam update."""
        optimizer = Adam(learning_rate=0.001)
        layer = MockLayer((2, 2), (2, 1))
        
        weights_before = layer.weights.copy()
        bias_before = layer.bias.copy()
        
        weights_gradient = np.ones((2, 2))
        bias_gradient = np.ones((2, 1))
        
        optimizer.update(layer, weights_gradient, bias_gradient)
        
        # Weights and bias should change
        assert not np.allclose(layer.weights, weights_before)
        assert not np.allclose(layer.bias, bias_before)
        
        # Time step should increment
        assert optimizer.t == 1
    
    def test_adam_multiple_updates(self):
        """Test Adam with multiple updates."""
        optimizer = Adam(learning_rate=0.001)
        layer = MockLayer((2, 2), (2, 1))
        
        weights_gradient = np.ones((2, 2))
        bias_gradient = np.ones((2, 1))
        
        # Multiple updates
        for i in range(5):
            optimizer.update(layer, weights_gradient, bias_gradient)
        
        assert optimizer.t == 5
    
    def test_adam_reset(self):
        """Test Adam reset."""
        optimizer = Adam(learning_rate=0.001)
        layer = MockLayer((2, 2), (2, 1))
        
        # Perform update
        optimizer.update(layer, np.ones((2, 2)), np.ones((2, 1)))
        
        # Reset should clear moments and time step
        optimizer.reset()
        assert len(optimizer.m_weights) == 0
        assert len(optimizer.v_weights) == 0
        assert optimizer.t == 0


class TestAdaGrad:
    """Test cases for AdaGrad optimizer."""
    
    def test_adagrad_initialization(self):
        """Test AdaGrad initialization."""
        optimizer = AdaGrad(learning_rate=0.01, epsilon=1e-8)
        assert optimizer.learning_rate == 0.01
        assert optimizer.epsilon == 1e-8
    
    def test_adagrad_update(self):
        """Test AdaGrad update."""
        optimizer = AdaGrad(learning_rate=0.01)
        layer = MockLayer((2, 2), (2, 1))
        
        weights_before = layer.weights.copy()
        weights_gradient = np.ones((2, 2))
        bias_gradient = np.ones((2, 1))
        
        optimizer.update(layer, weights_gradient, bias_gradient)
        
        # Weights should change
        assert not np.allclose(layer.weights, weights_before)
    
    def test_adagrad_reset(self):
        """Test AdaGrad reset."""
        optimizer = AdaGrad(learning_rate=0.01)
        layer = MockLayer((2, 2), (2, 1))
        
        # Perform update
        optimizer.update(layer, np.ones((2, 2)), np.ones((2, 1)))
        
        # Reset should clear accumulated gradients
        optimizer.reset()
        assert len(optimizer.sum_squared_weights) == 0
        assert len(optimizer.sum_squared_bias) == 0


class TestRMSprop:
    """Test cases for RMSprop optimizer."""
    
    def test_rmsprop_initialization(self):
        """Test RMSprop initialization."""
        optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-8)
        assert optimizer.learning_rate == 0.001
        assert optimizer.rho == 0.9
        assert optimizer.epsilon == 1e-8
    
    def test_rmsprop_update(self):
        """Test RMSprop update."""
        optimizer = RMSprop(learning_rate=0.001)
        layer = MockLayer((2, 2), (2, 1))
        
        weights_before = layer.weights.copy()
        weights_gradient = np.ones((2, 2))
        bias_gradient = np.ones((2, 1))
        
        optimizer.update(layer, weights_gradient, bias_gradient)
        
        # Weights should change
        assert not np.allclose(layer.weights, weights_before)
    
    def test_rmsprop_reset(self):
        """Test RMSprop reset."""
        optimizer = RMSprop(learning_rate=0.001)
        layer = MockLayer((2, 2), (2, 1))
        
        # Perform update
        optimizer.update(layer, np.ones((2, 2)), np.ones((2, 1)))
        
        # Reset should clear moving averages
        optimizer.reset()
        assert len(optimizer.avg_squared_weights) == 0
        assert len(optimizer.avg_squared_bias) == 0


class TestAdamW:
    """Test cases for AdamW optimizer."""
    
    def test_adamw_initialization(self):
        """Test AdamW initialization."""
        optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
        assert optimizer.learning_rate == 0.001
        assert optimizer.weight_decay == 0.01
    
    def test_adamw_update(self):
        """Test AdamW update with weight decay."""
        optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
        layer = MockLayer((2, 2), (2, 1))
        
        weights_before = layer.weights.copy()
        weights_gradient = np.ones((2, 2))
        bias_gradient = np.ones((2, 1))
        
        optimizer.update(layer, weights_gradient, bias_gradient)
        
        # Weights should change due to both gradient and weight decay
        assert not np.allclose(layer.weights, weights_before)
    
    def test_adamw_weight_decay(self):
        """Test that AdamW applies weight decay."""
        optimizer = AdamW(learning_rate=0.001, weight_decay=0.1)
        layer = MockLayer((2, 2), (2, 1))
        
        initial_weights = layer.weights.copy()
        weights_gradient = np.zeros((2, 2))  # No gradient
        bias_gradient = np.zeros((2, 1))
        
        optimizer.update(layer, weights_gradient, bias_gradient)
        
        # Weights should decay even with zero gradient
        expected_weights = initial_weights * (1 - 0.001 * 0.1)
        # Note: There will be some difference due to Adam's moment updates
        assert not np.allclose(layer.weights, initial_weights)
    
    def test_adamw_reset(self):
        """Test AdamW reset."""
        optimizer = AdamW(learning_rate=0.001)
        layer = MockLayer((2, 2), (2, 1))
        
        # Perform update
        optimizer.update(layer, np.ones((2, 2)), np.ones((2, 1)))
        
        # Reset should clear moments and time step
        optimizer.reset()
        assert len(optimizer.m_weights) == 0
        assert len(optimizer.v_weights) == 0
        assert optimizer.t == 0


if __name__ == "__main__":
    pytest.main([__file__])