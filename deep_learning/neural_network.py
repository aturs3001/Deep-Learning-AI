"""
Main Neural Network class that combines all components.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any, Union
from .layers import Layer, Dense, Dropout, BatchNormalization
from .activation import ACTIVATION_LAYERS
from .optimizers import Optimizer, OPTIMIZERS
from .loss import Loss, LOSS_FUNCTIONS
import pickle
import json


class NeuralNetwork:
    """Complete Neural Network implementation."""
    
    def __init__(self, loss: Union[str, Loss], optimizer: Union[str, Optimizer] = 'adam'):
        """
        Initialize neural network.
        
        Args:
            loss: Loss function (string or Loss instance)
            optimizer: Optimizer (string or Optimizer instance)
        """
        self.layers = []
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        
        # Set loss function
        if isinstance(loss, str):
            if loss not in LOSS_FUNCTIONS:
                raise ValueError(f"Unknown loss function: {loss}")
            self.loss_fn = LOSS_FUNCTIONS[loss]()
        else:
            self.loss_fn = loss
        
        # Set optimizer
        if isinstance(optimizer, str):
            if optimizer not in OPTIMIZERS:
                raise ValueError(f"Unknown optimizer: {optimizer}")
            self.optimizer = OPTIMIZERS[optimizer]()
        else:
            self.optimizer = optimizer
    
    def add(self, layer: Layer):
        """Add a layer to the network."""
        self.layers.append(layer)
    
    def add_dense(self, units: int, activation: str = 'relu', input_size: Optional[int] = None):
        """
        Add a dense layer with activation.
        
        Args:
            units: Number of neurons
            activation: Activation function name
            input_size: Input size (only needed for first layer)
        """
        # Determine input size
        if input_size is None:
            if len(self.layers) == 0:
                raise ValueError("Input size must be specified for the first layer")
            # Get output size from previous dense layer (skip activation layers)
            for layer in reversed(self.layers):
                if hasattr(layer, 'weights'):
                    input_size = layer.weights.shape[0]
                    break
            if input_size is None:
                raise ValueError("Cannot determine input size from previous layer")
        
        # Add dense layer
        dense_layer = Dense(input_size, units)
        self.add(dense_layer)
        
        # Add activation layer
        if activation in ACTIVATION_LAYERS:
            activation_layer = ACTIVATION_LAYERS[activation]()
            self.add(activation_layer)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def add_dropout(self, rate: float = 0.5):
        """Add a dropout layer."""
        dropout_layer = Dropout(rate)
        self.add(dropout_layer)
    
    def add_batch_norm(self, input_size: Optional[int] = None):
        """Add a batch normalization layer."""
        if input_size is None:
            if len(self.layers) == 0:
                raise ValueError("Input size must be specified for the first layer")
            # Get output size from previous dense layer (skip activation layers)
            for layer in reversed(self.layers):
                if hasattr(layer, 'weights'):
                    input_size = layer.weights.shape[0]
                    break
            if input_size is None:
                raise ValueError("Cannot determine input size from previous layer")
        
        batch_norm_layer = BatchNormalization(input_size)
        self.add(batch_norm_layer)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Backward pass through the network."""
        # Calculate initial gradient from loss function
        gradient = self.loss_fn.backward(y_pred, y_true)
        
        # Backpropagate through layers
        for layer in reversed(self.layers):
            if hasattr(layer, 'weights'):  # Dense layer
                weights_gradient = np.dot(gradient, layer.input.T)
                # Ensure bias gradient has correct shape by summing across batch dimension
                bias_gradient = np.sum(gradient, axis=1, keepdims=True)
                input_gradient = np.dot(layer.weights.T, gradient)
                
                # Update using optimizer
                self.optimizer.update(layer, weights_gradient, bias_gradient)
                gradient = input_gradient
            else:
                # Other layers (activation, dropout, etc.)
                gradient = layer.backward(gradient, 0)  # learning_rate not used
    
    def train_batch(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train on a single batch."""
        # Forward pass
        y_pred = self.forward(X)
        
        # Calculate loss
        loss = self.loss_fn.forward(y_pred, y)
        
        # Backward pass
        self.backward(y_pred, y)
        
        return loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        # Set layers to inference mode
        self.set_training(False)
        
        # Forward pass
        predictions = self.forward(X)
        
        # Set layers back to training mode
        self.set_training(True)
        
        return predictions
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate the model on test data."""
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate loss
        loss = self.loss_fn.forward(y_pred, y)
        
        # Calculate accuracy
        if y.shape[0] == 1:  # Binary classification
            accuracy = np.mean((y_pred > 0.5) == y)
        else:  # Multi-class classification
            accuracy = np.mean(np.argmax(y_pred, axis=0) == np.argmax(y, axis=0))
        
        return loss, accuracy
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
            batch_size: int = 32, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            verbose: bool = True, shuffle: bool = True) -> Dict[str, List[float]]:
        """
        Train the neural network.
        
        Args:
            X: Training data
            y: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_data: Tuple of (X_val, y_val) for validation
            verbose: Whether to print training progress
            shuffle: Whether to shuffle data each epoch
            
        Returns:
            Training history dictionary
        """
        # Ensure data is in correct format (features, samples)
        if len(X.shape) == 2 and X.shape[0] > X.shape[1]:
            X = X.T  # Transpose if more rows than columns
        if len(y.shape) == 2 and y.shape[0] > y.shape[1]:
            y = y.T  # Transpose if more rows than columns
        if len(y.shape) == 1:
            y = y.reshape(1, -1)  # Reshape 1D to (1, samples)
            
        n_samples = X.shape[1]
        n_batches = max(1, n_samples // batch_size)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Shuffle data if requested
            if shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[:, indices]
                y_shuffled = y[:, indices]
            else:
                X_shuffled, y_shuffled = X, y
            
            # Training batches
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[:, start_idx:end_idx]
                y_batch = y_shuffled[:, start_idx:end_idx]
                
                batch_loss = self.train_batch(X_batch, y_batch)
                epoch_loss += batch_loss
            
            # Average loss for the epoch
            avg_loss = epoch_loss / n_batches
            self.training_history['loss'].append(avg_loss)
            
            # Calculate training accuracy
            train_loss, train_acc = self.evaluate(X, y)
            self.training_history['accuracy'].append(train_acc)
            
            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                # Ensure validation data is in correct format too
                if len(X_val.shape) == 2 and X_val.shape[0] > X_val.shape[1]:
                    X_val = X_val.T
                if len(y_val.shape) == 2 and y_val.shape[0] > y_val.shape[1]:
                    y_val = y_val.T
                if len(y_val.shape) == 1:
                    y_val = y_val.reshape(1, -1)
                    
                val_loss, val_acc = self.evaluate(X_val, y_val)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_acc)
            
            # Print progress
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}")
                if validation_data is not None:
                    print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
                print()
        
        return self.training_history
    
    def set_training(self, training: bool):
        """Set training mode for all layers."""
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(training)
    
    def plot_history(self, figsize: Tuple[int, int] = (12, 4)):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot loss
        axes[0].plot(self.training_history['loss'], label='Training Loss')
        if self.training_history['val_loss']:
            axes[0].plot(self.training_history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        axes[1].plot(self.training_history['accuracy'], label='Training Accuracy')
        if self.training_history['val_accuracy']:
            axes[1].plot(self.training_history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save(self, filepath: str):
        """Save the model to a file."""
        model_data = {
            'layers': [],
            'training_history': self.training_history,
            'loss_fn': type(self.loss_fn).__name__,
            'optimizer': type(self.optimizer).__name__
        }
        
        # Save layer information
        for layer in self.layers:
            layer_data = {
                'type': type(layer).__name__,
                'params': {}
            }
            
            if hasattr(layer, 'weights'):
                layer_data['params']['weights'] = layer.weights
                layer_data['params']['bias'] = layer.bias
            
            if hasattr(layer, 'rate'):  # Dropout
                layer_data['params']['rate'] = layer.rate
            
            if hasattr(layer, 'gamma'):  # BatchNorm
                layer_data['params']['gamma'] = layer.gamma
                layer_data['params']['beta'] = layer.beta
                layer_data['params']['running_mean'] = layer.running_mean
                layer_data['params']['running_var'] = layer.running_var
            
            model_data['layers'].append(layer_data)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """Load a model from a file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore training history
        self.training_history = model_data['training_history']
        
        # Restore layers
        self.layers = []
        for layer_data in model_data['layers']:
            layer_type = layer_data['type']
            params = layer_data['params']
            
            if layer_type == 'Dense':
                layer = Dense(params['weights'].shape[1], params['weights'].shape[0])
                layer.weights = params['weights']
                layer.bias = params['bias']
            elif layer_type == 'Dropout':
                layer = Dropout(params['rate'])
            elif layer_type == 'BatchNormalization':
                layer = BatchNormalization(params['gamma'].shape[0])
                layer.gamma = params['gamma']
                layer.beta = params['beta']
                layer.running_mean = params['running_mean']
                layer.running_var = params['running_var']
            else:
                # Activation layers
                layer = ACTIVATION_LAYERS[layer_type.lower()]()
            
            self.layers.append(layer)
    
    def summary(self):
        """Print a summary of the network architecture."""
        print("Neural Network Summary")
        print("=" * 50)
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_name = type(layer).__name__
            
            if hasattr(layer, 'weights'):
                params = layer.weights.size + layer.bias.size
                output_shape = layer.weights.shape[0]
                print(f"Layer {i+1}: {layer_name} - Output: {output_shape}, Params: {params}")
                total_params += params
            else:
                print(f"Layer {i+1}: {layer_name}")
        
        print("=" * 50)
        print(f"Total Parameters: {total_params}")
        print(f"Loss Function: {type(self.loss_fn).__name__}")
        print(f"Optimizer: {type(self.optimizer).__name__}")
        print("=" * 50)