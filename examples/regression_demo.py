"""
Regression demo using the Deep Learning Framework.

This example demonstrates linear and non-linear regression
using synthetic datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os

# Add parent directory to path to import deep_learning
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deep_learning import NeuralNetwork


def create_linear_regression_data():
    """Create a linear regression dataset."""
    X, y = make_regression(
        n_samples=1000,
        n_features=1,
        noise=10,
        random_state=42
    )
    
    # Reshape for our framework
    X = X.T  # Shape: (n_features, n_samples)
    y = y.reshape(1, -1)  # Shape: (1, n_samples)
    
    return X, y


def create_polynomial_data():
    """Create polynomial regression data."""
    np.random.seed(42)
    x = np.linspace(-3, 3, 1000)
    y = 0.5 * x**3 - 2 * x**2 + x + 1 + np.random.normal(0, 2, len(x))
    
    X = x.reshape(1, -1)  # Shape: (1, n_samples)
    y = y.reshape(1, -1)  # Shape: (1, n_samples)
    
    return X, y


def create_sinusoidal_data():
    """Create sinusoidal regression data."""
    np.random.seed(42)
    x = np.linspace(0, 4*np.pi, 1000)
    y = np.sin(x) + 0.5 * np.cos(2*x) + np.random.normal(0, 0.1, len(x))
    
    X = x.reshape(1, -1)  # Shape: (1, n_samples)
    y = y.reshape(1, -1)  # Shape: (1, n_samples)
    
    return X, y


def create_multi_feature_data():
    """Create multi-feature regression data."""
    X, y = make_regression(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=42
    )
    
    # Reshape for our framework
    X = X.T  # Shape: (n_features, n_samples)
    y = y.reshape(1, -1)  # Shape: (1, n_samples)
    
    return X, y


def plot_regression_results(X, y, model, title="Regression Results"):
    """Plot regression results for 1D data."""
    if X.shape[0] != 1:
        print(f"Cannot plot {X.shape[0]}D data")
        return
    
    # Generate smooth curve for plotting
    x_plot = np.linspace(X.min(), X.max(), 300).reshape(1, -1)
    y_pred_plot = model.predict(x_plot)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X[0, :], y[0, :], alpha=0.6, label='Data Points')
    plt.plot(x_plot[0, :], y_pred_plot[0, :], 'r-', linewidth=2, label='Prediction')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def linear_regression_demo():
    """Demonstrate linear regression."""
    print("=== Linear Regression Demo ===")
    
    # Create data
    X, y = create_linear_regression_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X.T, y.T, test_size=0.2, random_state=42
    )
    X_train, X_test = X_train.T, X_test.T
    y_train, y_test = y_train.T, y_test.T
    
    # Standardize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train.T).T
    X_test_scaled = scaler_X.transform(X_test.T).T
    y_train_scaled = scaler_y.fit_transform(y_train.T).T
    y_test_scaled = scaler_y.transform(y_test.T).T
    
    # Create and train model
    model = NeuralNetwork(loss='mse', optimizer='adam')
    model.add_dense(units=8, activation='relu', input_size=1)
    model.add_dense(units=4, activation='relu')
    model.add_dense(units=1, activation='linear')
    
    print("Model Architecture:")
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=100,
        batch_size=32,
        validation_data=(X_test_scaled, y_test_scaled),
        verbose=True
    )
    
    # Make predictions
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.T).T
    
    # Evaluate
    mse = mean_squared_error(y_test[0, :], y_pred[0, :])
    mae = mean_absolute_error(y_test[0, :], y_pred[0, :])
    r2 = r2_score(y_test[0, :], y_pred[0, :])
    
    print(f"\nTest MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot training history
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Plot predictions vs actual
    plt.subplot(1, 3, 2)
    plt.scatter(y_test[0, :], y_pred[0, :], alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predictions vs Actual')
    plt.grid(True)
    
    # Plot data and fit
    plt.subplot(1, 3, 3)
    # Generate smooth predictions for plotting
    X_plot = np.linspace(X_test.min(), X_test.max(), 200).reshape(1, -1)
    X_plot_scaled = scaler_X.transform(X_plot.T).T
    y_plot_scaled = model.predict(X_plot_scaled)
    y_plot = scaler_y.inverse_transform(y_plot_scaled.T).T
    
    plt.scatter(X_test[0, :], y_test[0, :], alpha=0.6, label='Test Data')
    plt.plot(X_plot[0, :], y_plot[0, :], 'r-', linewidth=2, label='Model Fit')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model, history


def polynomial_regression_demo():
    """Demonstrate polynomial regression."""
    print("\n=== Polynomial Regression Demo ===")
    
    # Create data
    X, y = create_polynomial_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X.T, y.T, test_size=0.2, random_state=42
    )
    X_train, X_test = X_train.T, X_test.T
    y_train, y_test = y_train.T, y_test.T
    
    # Standardize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train.T).T
    X_test_scaled = scaler_X.transform(X_test.T).T
    y_train_scaled = scaler_y.fit_transform(y_train.T).T
    y_test_scaled = scaler_y.transform(y_test.T).T
    
    # Create and train model with more capacity for non-linear fitting
    model = NeuralNetwork(loss='mse', optimizer='adam')
    model.add_dense(units=20, activation='relu', input_size=1)
    model.add_dropout(rate=0.2)
    model.add_dense(units=15, activation='relu')
    model.add_dropout(rate=0.2)
    model.add_dense(units=10, activation='relu')
    model.add_dense(units=1, activation='linear')
    
    print("Model Architecture:")
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=200,
        batch_size=32,
        validation_data=(X_test_scaled, y_test_scaled),
        verbose=True
    )
    
    # Make predictions
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.T).T
    
    # Evaluate
    mse = mean_squared_error(y_test[0, :], y_pred[0, :])
    mae = mean_absolute_error(y_test[0, :], y_pred[0, :])
    r2 = r2_score(y_test[0, :], y_pred[0, :])
    
    print(f"\nTest MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot training history
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Plot predictions vs actual
    plt.subplot(1, 3, 2)
    plt.scatter(y_test[0, :], y_pred[0, :], alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predictions vs Actual')
    plt.grid(True)
    
    # Plot polynomial fit
    plt.subplot(1, 3, 3)
    # Sort data for better plotting
    sort_idx = np.argsort(X_test[0, :])
    plt.scatter(X_test[0, sort_idx], y_test[0, sort_idx], alpha=0.6, label='Test Data')
    
    # Generate smooth predictions
    X_plot = np.linspace(X_test.min(), X_test.max(), 300).reshape(1, -1)
    X_plot_scaled = scaler_X.transform(X_plot.T).T
    y_plot_scaled = model.predict(X_plot_scaled)
    y_plot = scaler_y.inverse_transform(y_plot_scaled.T).T
    
    plt.plot(X_plot[0, :], y_plot[0, :], 'r-', linewidth=2, label='Model Fit')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Polynomial Regression Fit')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model, history


def sinusoidal_regression_demo():
    """Demonstrate sinusoidal regression."""
    print("\n=== Sinusoidal Regression Demo ===")
    
    # Create data
    X, y = create_sinusoidal_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X.T, y.T, test_size=0.2, random_state=42
    )
    X_train, X_test = X_train.T, X_test.T
    y_train, y_test = y_train.T, y_test.T
    
    # Standardize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train.T).T
    X_test_scaled = scaler_X.transform(X_test.T).T
    y_train_scaled = scaler_y.fit_transform(y_train.T).T
    y_test_scaled = scaler_y.transform(y_test.T).T
    
    # Create and train model for complex oscillatory pattern
    model = NeuralNetwork(loss='mse', optimizer='adam')
    model.add_dense(units=50, activation='tanh', input_size=1)  # tanh works well for oscillatory functions
    model.add_dropout(rate=0.3)
    model.add_dense(units=30, activation='tanh')
    model.add_dropout(rate=0.3)
    model.add_dense(units=20, activation='relu')
    model.add_dense(units=1, activation='linear')
    
    print("Model Architecture:")
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=300,
        batch_size=32,
        validation_data=(X_test_scaled, y_test_scaled),
        verbose=True
    )
    
    # Make predictions
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.T).T
    
    # Evaluate
    mse = mean_squared_error(y_test[0, :], y_pred[0, :])
    mae = mean_absolute_error(y_test[0, :], y_pred[0, :])
    r2 = r2_score(y_test[0, :], y_pred[0, :])
    
    print(f"\nTest MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot training history
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Plot predictions vs actual
    plt.subplot(1, 3, 2)
    plt.scatter(y_test[0, :], y_pred[0, :], alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predictions vs Actual')
    plt.grid(True)
    
    # Plot sinusoidal fit
    plt.subplot(1, 3, 3)
    # Sort data for better plotting
    sort_idx = np.argsort(X_test[0, :])
    plt.scatter(X_test[0, sort_idx], y_test[0, sort_idx], alpha=0.6, label='Test Data')
    
    # Generate smooth predictions
    X_plot = np.linspace(X_test.min(), X_test.max(), 500).reshape(1, -1)
    X_plot_scaled = scaler_X.transform(X_plot.T).T
    y_plot_scaled = model.predict(X_plot_scaled)
    y_plot = scaler_y.inverse_transform(y_plot_scaled.T).T
    
    plt.plot(X_plot[0, :], y_plot[0, :], 'r-', linewidth=2, label='Model Fit')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Sinusoidal Regression Fit')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model, history


def multi_feature_regression_demo():
    """Demonstrate multi-feature regression."""
    print("\n=== Multi-Feature Regression Demo ===")
    
    # Create data
    X, y = create_multi_feature_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X.T, y.T, test_size=0.2, random_state=42
    )
    X_train, X_test = X_train.T, X_test.T
    y_train, y_test = y_train.T, y_test.T
    
    # Standardize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train.T).T
    X_test_scaled = scaler_X.transform(X_test.T).T
    y_train_scaled = scaler_y.fit_transform(y_train.T).T
    y_test_scaled = scaler_y.transform(y_test.T).T
    
    # Create and train model
    model = NeuralNetwork(loss='mse', optimizer='adam')
    model.add_dense(units=25, activation='relu', input_size=5)
    model.add_batch_norm()
    model.add_dropout(rate=0.3)
    model.add_dense(units=15, activation='relu')
    model.add_dropout(rate=0.2)
    model.add_dense(units=8, activation='relu')
    model.add_dense(units=1, activation='linear')
    
    print("Model Architecture:")
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=150,
        batch_size=32,
        validation_data=(X_test_scaled, y_test_scaled),
        verbose=True
    )
    
    # Make predictions
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.T).T
    
    # Evaluate
    mse = mean_squared_error(y_test[0, :], y_pred[0, :])
    mae = mean_absolute_error(y_test[0, :], y_pred[0, :])
    r2 = r2_score(y_test[0, :], y_pred[0, :])
    
    print(f"\nTest MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot training history
    plt.subplot(2, 3, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Plot predictions vs actual
    plt.subplot(2, 3, 2)
    plt.scatter(y_test[0, :], y_pred[0, :], alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predictions vs Actual')
    plt.grid(True)
    
    # Plot residuals
    plt.subplot(2, 3, 3)
    residuals = y_test[0, :] - y_pred[0, :]
    plt.scatter(y_pred[0, :], residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    
    # Plot feature importance (approximate using weight magnitudes)
    plt.subplot(2, 3, 4)
    first_layer_weights = model.layers[0].weights
    feature_importance = np.mean(np.abs(first_layer_weights), axis=0)
    feature_names = [f'Feature {i+1}' for i in range(len(feature_importance))]
    
    plt.bar(feature_names, feature_importance)
    plt.title('Approximate Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Average |Weight|')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # Plot learning curves
    plt.subplot(2, 3, 5)
    epochs = range(1, len(history['loss']) + 1)
    plt.plot(epochs, history['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot error distribution
    plt.subplot(2, 3, 6)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return model, history


def compare_optimizers_demo():
    """Compare different optimizers on the same regression task."""
    print("\n=== Optimizer Comparison Demo ===")
    
    # Create data
    X, y = create_polynomial_data()
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X.T, y.T, test_size=0.2, random_state=42
    )
    X_train, X_test = X_train.T, X_test.T
    y_train, y_test = y_train.T, y_test.T
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train.T).T
    X_test_scaled = scaler_X.transform(X_test.T).T
    y_train_scaled = scaler_y.fit_transform(y_train.T).T
    y_test_scaled = scaler_y.transform(y_test.T).T
    
    # Define optimizers to compare
    optimizers = ['sgd', 'adam', 'rmsprop', 'adagrad']
    results = {}
    
    plt.figure(figsize=(15, 10))
    
    for i, opt_name in enumerate(optimizers):
        print(f"\nTraining with {opt_name.upper()} optimizer...")
        
        # Create model
        model = NeuralNetwork(loss='mse', optimizer=opt_name)
        model.add_dense(units=20, activation='relu', input_size=1)
        model.add_dense(units=15, activation='relu')
        model.add_dense(units=10, activation='relu')
        model.add_dense(units=1, activation='linear')
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=100,
            batch_size=32,
            validation_data=(X_test_scaled, y_test_scaled),
            verbose=False
        )
        
        # Evaluate
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.T).T
        mse = mean_squared_error(y_test[0, :], y_pred[0, :])
        r2 = r2_score(y_test[0, :], y_pred[0, :])
        
        results[opt_name] = {
            'model': model,
            'history': history,
            'mse': mse,
            'r2': r2
        }
        
        print(f"{opt_name.upper()} - Test MSE: {mse:.4f}, Test R²: {r2:.4f}")
        
        # Plot training curves
        plt.subplot(2, 2, i+1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'{opt_name.upper()} Optimizer')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Summary comparison
    print("\n" + "="*50)
    print("OPTIMIZER COMPARISON SUMMARY")
    print("="*50)
    for opt_name, result in results.items():
        print(f"{opt_name.upper():10} - MSE: {result['mse']:8.4f}, R²: {result['r2']:6.4f}")
    
    # Find best optimizer
    best_opt = min(results.keys(), key=lambda x: results[x]['mse'])
    print(f"\nBest optimizer: {best_opt.upper()} (lowest MSE)")
    
    return results


def main():
    """Run all regression demos."""
    print("Deep Learning Framework - Regression Examples")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Run demos
    linear_model, linear_history = linear_regression_demo()
    poly_model, poly_history = polynomial_regression_demo()
    sin_model, sin_history = sinusoidal_regression_demo()
    multi_model, multi_history = multi_feature_regression_demo()
    optimizer_results = compare_optimizers_demo()
    
    print("\n" + "=" * 50)
    print("All regression demos completed successfully!")
    print("=" * 50)
    
    return {
        'linear': (linear_model, linear_history),
        'polynomial': (poly_model, poly_history),
        'sinusoidal': (sin_model, sin_history),
        'multi_feature': (multi_model, multi_history),
        'optimizer_comparison': optimizer_results
    }


if __name__ == "__main__":
    results = main()