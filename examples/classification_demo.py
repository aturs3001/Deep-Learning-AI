# ============================================================
# classification_demo.py
# Author: Aric Hurkman
# Date: 2025-05-27
# Copyright (c) 2023 Aric Hurkman
# License: All rights reserved.
# Disclaimer: This code is for Portfolio and Educational purposes only.
# ============================================================
# Description:
# Classification demo using the Deep Learning Framework.
# This example demonstrates binary and multi-class classification
# using synthetic datasets.
# It includes:
# - Binary classification with a simple dataset
# - Multi-class classification with a more complex dataset
# - Non-linear classification with circular data
# It visualizes the training history and decision boundaries.
# Requirements:
# - numpy
# - matplotlib
# - scikit-learn
# - deep_learning (custom framework)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# Add parent directory to path to import deep_learning
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deep_learning import NeuralNetwork


def create_binary_classification_data():
    """Create a binary classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert to binary (0, 1) and reshape for our framework
    y = (y + 1) // 2  # Convert from {-1, 1} to {0, 1}
    y = y.reshape(1, -1)  # Shape: (1, n_samples)
    X = X.T  # Shape: (n_features, n_samples)
    
    return X, y


def create_multi_class_data():
    """Create a multi-class classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert to one-hot encoding
    n_classes = len(np.unique(y))
    y_onehot = np.zeros((n_classes, len(y)))
    for i, label in enumerate(y):
        y_onehot[label, i] = 1
    
    X = X.T  # Shape: (n_features, n_samples)
    
    return X, y_onehot, y


def create_circular_data():
    """Create a circular dataset for non-linear classification."""
    X, y = make_circles(n_samples=1000, noise=0.1, factor=0.3, random_state=42)
    
    y = y.reshape(1, -1)  # Shape: (1, n_samples)
    X = X.T  # Shape: (n_features, n_samples)
    
    return X, y


def binary_classification_demo():
    """Demonstrate binary classification."""
    print("=== Binary Classification Demo ===")
    
    # Create data
    X, y = create_binary_classification_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X.T, y.T, test_size=0.2, random_state=42
    )
    X_train, X_test = X_train.T, X_test.T
    y_train, y_test = y_train.T, y_test.T
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.T).T
    X_test = scaler.transform(X_test.T).T
    
    # Create and train model
    model = NeuralNetwork(loss='binary_crossentropy', optimizer='adam')
    model.add_dense(units=10, activation='relu', input_size=2)
    model.add_dense(units=5, activation='relu')
    model.add_dense(units=1, activation='sigmoid')
    
    print("Model Architecture:")
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=True
    )
    
    # Evaluate
    train_loss, train_acc = model.evaluate(X_train, y_train)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot training history
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training History - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training History - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot decision boundary
    plt.subplot(1, 3, 3)
    # Create a mesh for decision boundary
    h = 0.02
    x_min, x_max = X_test[0, :].min() - 1, X_test[0, :].max() + 1
    y_min, y_max = X_test[1, :].min() - 1, X_test[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()].T
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    colors = ['red' if label == 0 else 'blue' for label in y_test[0, :]]
    plt.scatter(X_test[0, :], X_test[1, :], c=colors, edgecolors='black')
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Prediction')
    
    plt.tight_layout()
    plt.show()
    
    return model, history


def multi_class_classification_demo():
    """Demonstrate multi-class classification."""
    print("\n=== Multi-Class Classification Demo ===")
    
    # Create data
    X, y_onehot, y_labels = create_multi_class_data()
    
    # Split data
    X_train, X_test, y_train, y_test, y_labels_train, y_labels_test = train_test_split(
        X.T, y_onehot.T, y_labels, test_size=0.2, random_state=42
    )
    X_train, X_test = X_train.T, X_test.T
    y_train, y_test = y_train.T, y_test.T
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.T).T
    X_test = scaler.transform(X_test.T).T
    
    # Create and train model
    model = NeuralNetwork(loss='categorical_crossentropy', optimizer='adam')
    model.add_dense(units=15, activation='relu', input_size=2)
    model.add_dropout(rate=0.3)
    model.add_dense(units=10, activation='relu')
    model.add_dense(units=3, activation='softmax')  # 3 classes
    
    print("Model Architecture:")
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=True
    )
    
    # Evaluate
    train_loss, train_acc = model.evaluate(X_train, y_train)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Get predictions for detailed analysis
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=0)
    
    print("\nClassification Report:")
    print(classification_report(y_labels_test, y_pred_labels))
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot training history
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training History - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training History - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot decision boundary
    plt.subplot(1, 3, 3)
    h = 0.02
    x_min, x_max = X_test[0, :].min() - 1, X_test[0, :].max() + 1
    y_min, y_max = X_test[1, :].min() - 1, X_test[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()].T
    Z = model.predict(mesh_points)
    Z = np.argmax(Z, axis=0).reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='viridis')
    colors = plt.cm.viridis(y_labels_test / 2)  # 3 classes: 0, 1, 2
    plt.scatter(X_test[0, :], X_test[1, :], c=colors, edgecolors='black')
    plt.title('Multi-Class Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')
    
    plt.tight_layout()
    plt.show()
    
    return model, history


def circular_classification_demo():
    """Demonstrate non-linear classification with circular data."""
    print("\n=== Non-Linear Classification Demo (Circular Data) ===")
    
    # Create circular data
    X, y = create_circular_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X.T, y.T, test_size=0.2, random_state=42
    )
    X_train, X_test = X_train.T, X_test.T
    y_train, y_test = y_train.T, y_test.T
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.T).T
    X_test = scaler.transform(X_test.T).T
    
    # Create and train model with more complex architecture
    model = NeuralNetwork(loss='binary_crossentropy', optimizer='adam')
    model.add_dense(units=20, activation='relu', input_size=2)
    model.add_dropout(rate=0.2)
    model.add_dense(units=15, activation='relu')
    model.add_dropout(rate=0.2)
    model.add_dense(units=10, activation='relu')
    model.add_dense(units=1, activation='sigmoid')
    
    print("Model Architecture:")
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=True
    )
    
    # Evaluate
    train_loss, train_acc = model.evaluate(X_train, y_train)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot training history
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training History - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training History - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot decision boundary for circular data
    plt.subplot(1, 3, 3)
    h = 0.02
    x_min, x_max = X_test[0, :].min() - 1, X_test[0, :].max() + 1
    y_min, y_max = X_test[1, :].min() - 1, X_test[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()].T
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    colors = ['red' if label == 0 else 'blue' for label in y_test[0, :]]
    plt.scatter(X_test[0, :], X_test[1, :], c=colors, edgecolors='black')
    plt.title('Circular Data Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Prediction')
    
    plt.tight_layout()
    plt.show()
    
    return model, history


def main():
    """Run all classification demos."""
    print("Deep Learning Framework - Classification Examples")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Run demos
    binary_model, binary_history = binary_classification_demo()
    multi_model, multi_history = multi_class_classification_demo()
    circular_model, circular_history = circular_classification_demo()
    
    print("\n" + "=" * 50)
    print("All demos completed successfully!")
    print("=" * 50)
    
    return {
        'binary': (binary_model, binary_history),
        'multi_class': (multi_model, multi_history),
        'circular': (circular_model, circular_history)
    }


if __name__ == "__main__":
    results = main()