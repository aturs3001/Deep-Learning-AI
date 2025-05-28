"""
Tests for loss functions.
"""

import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path to import deep_learning
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deep_learning.loss import (
    MeanSquaredError, MeanAbsoluteError, BinaryCrossentropy,
    CategoricalCrossentropy, SparseCategoricalCrossentropy,
    Huber, LogCosh, Hinge, SquaredHinge, KLDivergence
)


class TestMeanSquaredError:
    """Test cases for Mean Squared Error loss."""
    
    def test_mse_forward(self):
        """Test MSE forward pass."""
        loss_fn = MeanSquaredError()
        y_pred = np.array([[1, 2], [3, 4]])
        y_true = np.array([[1, 1], [1, 1]])
        
        loss = loss_fn.forward(y_pred, y_true)
        expected_loss = np.mean((y_pred - y_true) ** 2)
        
        assert abs(loss - expected_loss) < 1e-10
    
    def test_mse_backward(self):
        """Test MSE backward pass."""
        loss_fn = MeanSquaredError()
        y_pred = np.array([[1, 2], [3, 4]])
        y_true = np.array([[1, 1], [1, 1]])
        
        gradient = loss_fn.backward(y_pred, y_true)
        expected_gradient = 2 * (y_pred - y_true) / y_pred.shape[1]
        
        np.testing.assert_array_almost_equal(gradient, expected_gradient)


class TestMeanAbsoluteError:
    """Test cases for Mean Absolute Error loss."""
    
    def test_mae_forward(self):
        """Test MAE forward pass."""
        loss_fn = MeanAbsoluteError()
        y_pred = np.array([[1, 2], [3, 4]])
        y_true = np.array([[1, 1], [1, 1]])
        
        loss = loss_fn.forward(y_pred, y_true)
        expected_loss = np.mean(np.abs(y_pred - y_true))
        
        assert abs(loss - expected_loss) < 1e-10
    
    def test_mae_backward(self):
        """Test MAE backward pass."""
        loss_fn = MeanAbsoluteError()
        y_pred = np.array([[1, 2], [3, 4]])
        y_true = np.array([[1, 1], [1, 1]])
        
        gradient = loss_fn.backward(y_pred, y_true)
        expected_gradient = np.sign(y_pred - y_true) / y_pred.shape[1]
        
        np.testing.assert_array_almost_equal(gradient, expected_gradient)


class TestBinaryCrossentropy:
    """Test cases for Binary Cross-entropy loss."""
    
    def test_binary_crossentropy_forward(self):
        """Test binary cross-entropy forward pass."""
        loss_fn = BinaryCrossentropy()
        y_pred = np.array([[0.8, 0.2], [0.7, 0.9]])
        y_true = np.array([[1, 0], [1, 1]])
        
        loss = loss_fn.forward(y_pred, y_true)
        
        # Loss should be positive
        assert loss > 0
    
    def test_binary_crossentropy_backward(self):
        """Test binary cross-entropy backward pass."""
        loss_fn = BinaryCrossentropy()
        y_pred = np.array([[0.8, 0.2], [0.7, 0.9]])
        y_true = np.array([[1, 0], [1, 1]])
        
        gradient = loss_fn.backward(y_pred, y_true)
        
        assert gradient.shape == y_pred.shape
    
    def test_binary_crossentropy_extreme_values(self):
        """Test binary cross-entropy with extreme values."""
        loss_fn = BinaryCrossentropy()
        
        # Test with values close to 0 and 1
        y_pred = np.array([[0.001, 0.999]])
        y_true = np.array([[0, 1]])
        
        loss = loss_fn.forward(y_pred, y_true)
        gradient = loss_fn.backward(y_pred, y_true)
        
        # Should not produce NaN or infinite values
        assert np.isfinite(loss)
        assert np.all(np.isfinite(gradient))


class TestCategoricalCrossentropy:
    """Test cases for Categorical Cross-entropy loss."""
    
    def test_categorical_crossentropy_forward(self):
        """Test categorical cross-entropy forward pass."""
        loss_fn = CategoricalCrossentropy()
        y_pred = np.array([[0.8, 0.7], [0.1, 0.2], [0.1, 0.1]])
        y_true = np.array([[1, 0], [0, 1], [0, 0]])
        
        loss = loss_fn.forward(y_pred, y_true)
        
        # Loss should be positive
        assert loss > 0
    
    def test_categorical_crossentropy_backward(self):
        """Test categorical cross-entropy backward pass."""
        loss_fn = CategoricalCrossentropy()
        y_pred = np.array([[0.8, 0.7], [0.1, 0.2], [0.1, 0.1]])
        y_true = np.array([[1, 0], [0, 1], [0, 0]])
        
        gradient = loss_fn.backward(y_pred, y_true)
        
        assert gradient.shape == y_pred.shape


class TestSparseCategoricalCrossentropy:
    """Test cases for Sparse Categorical Cross-entropy loss."""
    
    def test_sparse_categorical_crossentropy_forward(self):
        """Test sparse categorical cross-entropy forward pass."""
        loss_fn = SparseCategoricalCrossentropy()
        y_pred = np.array([[0.8, 0.7], [0.1, 0.2], [0.1, 0.1]])
        y_true = np.array([[0, 1]])  # Sparse labels (class indices)
        
        loss = loss_fn.forward(y_pred, y_true)
        
        # Loss should be positive
        assert loss > 0
    
    def test_sparse_categorical_crossentropy_backward(self):
        """Test sparse categorical cross-entropy backward pass."""
        loss_fn = SparseCategoricalCrossentropy()
        y_pred = np.array([[0.8, 0.7], [0.1, 0.2], [0.1, 0.1]])
        y_true = np.array([[0, 1]])  # Sparse labels
        
        gradient = loss_fn.backward(y_pred, y_true)
        
        assert gradient.shape == y_pred.shape


class TestHuber:
    """Test cases for Huber loss."""
    
    def test_huber_forward(self):
        """Test Huber loss forward pass."""
        loss_fn = Huber(delta=1.0)
        y_pred = np.array([[1, 2], [3, 4]])
        y_true = np.array([[1, 1], [1, 1]])
        
        loss = loss_fn.forward(y_pred, y_true)
        
        # Loss should be positive
        assert loss >= 0
    
    def test_huber_backward(self):
        """Test Huber loss backward pass."""
        loss_fn = Huber(delta=1.0)
        y_pred = np.array([[1, 2], [3, 4]])
        y_true = np.array([[1, 1], [1, 1]])
        
        gradient = loss_fn.backward(y_pred, y_true)
        
        assert gradient.shape == y_pred.shape
    
    def test_huber_small_errors(self):
        """Test Huber loss with small errors (quadratic region)."""
        loss_fn = Huber(delta=2.0)
        y_pred = np.array([[1.5]])
        y_true = np.array([[1.0]])
        
        # Error = 0.5, which is < delta, so should be quadratic
        loss = loss_fn.forward(y_pred, y_true)
        expected_loss = 0.5 * 0.5 * 0.5  # 0.5 * error^2
        
        assert abs(loss - expected_loss) < 1e-10
    
    def test_huber_large_errors(self):
        """Test Huber loss with large errors (linear region)."""
        loss_fn = Huber(delta=1.0)
        y_pred = np.array([[3.0]])
        y_true = np.array([[1.0]])
        
        # Error = 2.0, which is > delta, so should be linear
        loss = loss_fn.forward(y_pred, y_true)
        expected_loss = 1.0 * (2.0 - 0.5 * 1.0)  # delta * (|error| - 0.5 * delta)
        
        assert abs(loss - expected_loss) < 1e-10


class TestLogCosh:
    """Test cases for Log-Cosh loss."""
    
    def test_log_cosh_forward(self):
        """Test Log-Cosh loss forward pass."""
        loss_fn = LogCosh()
        y_pred = np.array([[1, 2], [3, 4]])
        y_true = np.array([[1, 1], [1, 1]])
        
        loss = loss_fn.forward(y_pred, y_true)
        
        # Loss should be positive
        assert loss >= 0
    
    def test_log_cosh_backward(self):
        """Test Log-Cosh loss backward pass."""
        loss_fn = LogCosh()
        y_pred = np.array([[1, 2], [3, 4]])
        y_true = np.array([[1, 1], [1, 1]])
        
        gradient = loss_fn.backward(y_pred, y_true)
        
        assert gradient.shape == y_pred.shape
    
    def test_log_cosh_zero_error(self):
        """Test Log-Cosh loss with zero error."""
        loss_fn = LogCosh()
        y_pred = np.array([[1, 2]])
        y_true = np.array([[1, 2]])
        
        loss = loss_fn.forward(y_pred, y_true)
        gradient = loss_fn.backward(y_pred, y_true)
        
        assert abs(loss - np.log(1)) < 1e-10  # log(cosh(0)) = log(1) = 0
        np.testing.assert_array_almost_equal(gradient, np.zeros_like(y_pred))


class TestHinge:
    """Test cases for Hinge loss."""
    
    def test_hinge_forward(self):
        """Test Hinge loss forward pass."""
        loss_fn = Hinge()
        y_pred = np.array([[0.8, -0.2]])
        y_true = np.array([[1, -1]])  # Binary labels: +1 or -1
        
        loss = loss_fn.forward(y_pred, y_true)
        
        # Loss should be non-negative
        assert loss >= 0
    
    def test_hinge_backward(self):
        """Test Hinge loss backward pass."""
        loss_fn = Hinge()
        y_pred = np.array([[0.8, -0.2]])
        y_true = np.array([[1, -1]])
        
        gradient = loss_fn.backward(y_pred, y_true)
        
        assert gradient.shape == y_pred.shape
    
    def test_hinge_correct_classification(self):
        """Test Hinge loss with correct classification (margin > 1)."""
        loss_fn = Hinge()
        y_pred = np.array([[2.0]])
        y_true = np.array([[1]])
        
        # y_true * y_pred = 2 > 1, so loss should be 0
        loss = loss_fn.forward(y_pred, y_true)
        gradient = loss_fn.backward(y_pred, y_true)
        
        assert abs(loss) < 1e-10
        assert abs(gradient[0, 0]) < 1e-10
    
    def test_hinge_incorrect_classification(self):
        """Test Hinge loss with incorrect classification."""
        loss_fn = Hinge()
        y_pred = np.array([[-0.5]])
        y_true = np.array([[1]])
        
        # y_true * y_pred = -0.5 < 1, so loss should be 1 - (-0.5) = 1.5
        loss = loss_fn.forward(y_pred, y_true)
        expected_loss = 1.5
        
        assert abs(loss - expected_loss) < 1e-10


class TestSquaredHinge:
    """Test cases for Squared Hinge loss."""
    
    def test_squared_hinge_forward(self):
        """Test Squared Hinge loss forward pass."""
        loss_fn = SquaredHinge()
        y_pred = np.array([[0.8, -0.2]])
        y_true = np.array([[1, -1]])
        
        loss = loss_fn.forward(y_pred, y_true)
        
        # Loss should be non-negative
        assert loss >= 0
    
    def test_squared_hinge_backward(self):
        """Test Squared Hinge loss backward pass."""
        loss_fn = SquaredHinge()
        y_pred = np.array([[0.8, -0.2]])
        y_true = np.array([[1, -1]])
        
        gradient = loss_fn.backward(y_pred, y_true)
        
        assert gradient.shape == y_pred.shape


class TestKLDivergence:
    """Test cases for KL Divergence loss."""
    
    def test_kl_divergence_forward(self):
        """Test KL Divergence forward pass."""
        loss_fn = KLDivergence()
        y_pred = np.array([[0.8, 0.7], [0.2, 0.3]])
        y_true = np.array([[0.9, 0.6], [0.1, 0.4]])
        
        loss = loss_fn.forward(y_pred, y_true)
        
        # Loss should be non-negative
        assert loss >= 0
    
    def test_kl_divergence_backward(self):
        """Test KL Divergence backward pass."""
        loss_fn = KLDivergence()
        y_pred = np.array([[0.8, 0.7], [0.2, 0.3]])
        y_true = np.array([[0.9, 0.6], [0.1, 0.4]])
        
        gradient = loss_fn.backward(y_pred, y_true)
        
        assert gradient.shape == y_pred.shape
    
    def test_kl_divergence_identical_distributions(self):
        """Test KL Divergence with identical distributions."""
        loss_fn = KLDivergence()
        y_pred = np.array([[0.6, 0.5], [0.4, 0.5]])
        y_true = y_pred.copy()
        
        loss = loss_fn.forward(y_pred, y_true)
        
        # KL divergence should be 0 for identical distributions
        assert abs(loss) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])