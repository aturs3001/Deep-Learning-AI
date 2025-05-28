"""
Simplified Classification Demo using the Deep Learning Framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add parent directory to path to import deep_learning
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deep_learning import NeuralNetwork


def binary_classification_demo():
    """Demonstrate binary classification."""
    print("=== Binary Classification Demo ===")
    
    # Create data
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert to our framework format
    y = (y + 1) // 2  # Convert from {-1, 1} to {0, 1}
    y = y.reshape(1, -1)  # Shape: (1, n_samples)
    X = X.T  # Shape: (n_features, n_samples)
    
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
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=True
    )
    
    # Evaluate
    train_loss, train_acc = model.evaluate(X_train, y_train)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {train_acc:.4f}")
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
    
    # Plot data points
    plt.subplot(1, 3, 3)
    colors = ['red' if label == 0 else 'blue' for label in y_test[0, :]]
    plt.scatter(X_test[0, :], X_test[1, :], c=colors, edgecolors='black')
    plt.title('Test Data Classification')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model, history


def circular_classification_demo():
    """Demonstrate non-linear classification with circular data."""
    print("\n=== Circular Classification Demo ===")
    
    # Create circular data
    X, y = make_circles(n_samples=500, noise=0.1, factor=0.3, random_state=42)
    
    # Convert to our framework format
    y = y.reshape(1, -1)  # Shape: (1, n_samples)
    X = X.T  # Shape: (n_features, n_samples)
    
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
    print("\nTraining...")
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
    
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {train_acc:.4f}")
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
    
    # Plot circular data
    plt.subplot(1, 3, 3)
    colors = ['red' if label == 0 else 'blue' for label in y_test[0, :]]
    plt.scatter(X_test[0, :], X_test[1, :], c=colors, edgecolors='black')
    plt.title('Circular Data Classification')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model, history


def multi_class_demo():
    """Demonstrate multi-class classification."""
    print("\n=== Multi-Class Classification Demo ===")
    
    # Create data
    X, y = make_classification(
        n_samples=600,
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
    
    # Split data
    X_train, X_test, y_train, y_test, y_labels_train, y_labels_test = train_test_split(
        X.T, y_onehot.T, y, test_size=0.2, random_state=42
    )
    X_train, X_test = X_train.T, X_test.T
    y_train, y_test = y_train.T, y_test.T
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.T).T
    X_test = scaler.transform(X_test.T).T
    
    # Create and train model
    model = NeuralNetwork(loss='categorical_crossentropy', optimizer='adam')
    model.add_dense(units=20, activation='relu', input_size=2)  # Increased capacity
    model.add_dense(units=15, activation='relu')  # Added extra layer
    model.add_dropout(rate=0.2)  # Reduced dropout
    model.add_dense(units=10, activation='relu')
    model.add_dense(units=3, activation='softmax')  # 3 classes
    
    print("Model Architecture:")
    model.summary()
    
    # Train the model
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=120,  # Increased epochs
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=True
    )
    
    # Evaluate
    train_loss, train_acc = model.evaluate(X_train, y_train)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {train_acc:.4f}")
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
    
    # Plot multi-class data
    plt.subplot(1, 3, 3)
    colors = plt.cm.viridis(y_labels_test / 2)  # 3 classes: 0, 1, 2
    plt.scatter(X_test[0, :], X_test[1, :], c=colors, edgecolors='black')
    plt.title('Multi-Class Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')
    plt.grid(True)
    
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
    print("Running Binary Classification Demo...")
    binary_model, binary_history = binary_classification_demo()
    
    print("\nRunning Circular Classification Demo...")
    circular_model, circular_history = circular_classification_demo()
    
    print("\nRunning Multi-Class Classification Demo...")
    multi_model, multi_history = multi_class_demo()
    
    print("\n" + "=" * 50)
    print("All demos completed successfully!")
    print("=" * 50)
    
    return {
        'binary': (binary_model, binary_history),
        'circular': (circular_model, circular_history),
        'multi_class': (multi_model, multi_history)
    }


if __name__ == "__main__":
    results = main()