# Deep Learning Framework

A comprehensive deep learning framework built from scratch in Python, featuring neural networks, multiple activation functions, optimizers, and loss functions.

## Features

### Core Components
- **Neural Network Architecture**: Flexible network construction with easy layer addition
- **Layer Types**: Dense, Dropout, Batch Normalization
- **Activation Functions**: Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Swish, Softmax, Linear
- **Optimizers**: SGD, Adam, AdaGrad, RMSprop, AdamW
- **Loss Functions**: MSE, MAE, Binary/Categorical Cross-entropy, Huber, Hinge, and more

### Advanced Features
- Training history tracking
- Model saving/loading
- Batch normalization
- Dropout regularization
- Validation during training
- Comprehensive plotting utilities

## Installation

### From Source
```bash
git clone https://github.com/aturs3001/deep-learning-framework.git
cd deep-learning-framework
pip install -e .
```

### Requirements
```bash
pip install -r requirements.txt
```

## Quick Start

### Binary Classification
```python
from deep_learning import NeuralNetwork
import numpy as np

# Create sample data
X = np.random.randn(2, 1000)  # 2 features, 1000 samples
y = (X[0] + X[1] > 0).astype(int).reshape(1, -1)  # Binary labels

# Create and configure model
model = NeuralNetwork(loss='binary_crossentropy', optimizer='adam')
model.add_dense(units=10, activation='relu', input_size=2)
model.add_dense(units=5, activation='relu')
model.add_dense(units=1, activation='sigmoid')

# Train the model
history = model.fit(X, y, epochs=100, batch_size=32, verbose=True)

# Make predictions
predictions = model.predict(X)
```

### Multi-class Classification
```python
# Create model for 3-class classification
model = NeuralNetwork(loss='categorical_crossentropy', optimizer='adam')
model.add_dense(units=15, activation='relu', input_size=2)
model.add_dropout(rate=0.3)
model.add_dense(units=10, activation='relu')
model.add_dense(units=3, activation='softmax')  # 3 classes

# Train with validation data
history = model.fit(
    X_train, y_train,
    epochs=150,
    validation_data=(X_val, y_val),
    verbose=True
)
```

### Regression
```python
# Create regression model
model = NeuralNetwork(loss='mse', optimizer='adam')
model.add_dense(units=20, activation='relu', input_size=1)
model.add_dense(units=10, activation='relu')
model.add_dense(units=1, activation='linear')

# Train the model
history = model.fit(X, y, epochs=200, batch_size=32)

# Evaluate performance
loss, accuracy = model.evaluate(X_test, y_test)
```

## Advanced Usage

### Custom Architecture with Regularization
```python
model = NeuralNetwork(loss='categorical_crossentropy', optimizer='adam')
model.add_dense(units=64, activation='relu', input_size=10)
model.add_batch_norm()  # Batch normalization
model.add_dropout(rate=0.5)  # Dropout regularization
model.add_dense(units=32, activation='relu')
model.add_dropout(rate=0.3)
model.add_dense(units=5, activation='softmax')
```

### Using Different Optimizers
```python
# SGD with momentum
model = NeuralNetwork(loss='mse', optimizer='sgd')

# Adam optimizer (default parameters)
model = NeuralNetwork(loss='mse', optimizer='adam')

# Custom optimizer
from deep_learning.optimizers import Adam
custom_optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
model = NeuralNetwork(loss='mse', optimizer=custom_optimizer)
```

### Model Persistence
```python
# Save trained model
model.save('my_model.pkl')

# Load model
model = NeuralNetwork(loss='mse', optimizer='adam')
model.load('my_model.pkl')
```

### Training Visualization
```python
# Plot training history
model.plot_history()

# Access training metrics
print("Training loss:", history['loss'])
print("Validation accuracy:", history['val_accuracy'])
```

## Examples

The framework includes comprehensive examples:

### Classification Examples
```bash
python examples/classification_demo.py
```
- Binary classification
- Multi-class classification  
- Non-linear classification (circular data)

### Regression Examples
```bash
python examples/regression_demo.py
```
- Linear regression
- Polynomial regression
- Sinusoidal regression
- Multi-feature regression
- Optimizer comparison

## API Reference

### NeuralNetwork Class
```python
class NeuralNetwork:
    def __init__(self, loss, optimizer='adam')
    def add_dense(self, units, activation='relu', input_size=None)
    def add_dropout(self, rate=0.5)
    def add_batch_norm(self, input_size=None)
    def fit(self, X, y, epochs=100, batch_size=32, validation_data=None)
    def predict(self, X)
    def evaluate(self, X, y)
    def save(self, filepath)
    def load(self, filepath)
```

### Available Activation Functions
- `'sigmoid'` - Sigmoid activation
- `'tanh'` - Hyperbolic tangent  
- `'relu'` - Rectified Linear Unit
- `'leaky_relu'` - Leaky ReLU
- `'elu'` - Exponential Linear Unit
- `'swish'` - Swish activation
- `'softmax'` - Softmax (for classification)
- `'linear'` - Linear activation

### Available Loss Functions
- `'mse'` - Mean Squared Error
- `'mae'` - Mean Absolute Error
- `'binary_crossentropy'` - Binary Cross-entropy
- `'categorical_crossentropy'` - Categorical Cross-entropy
- `'sparse_categorical_crossentropy'` - Sparse Categorical Cross-entropy
- `'huber'` - Huber loss
- `'hinge'` - Hinge loss

### Available Optimizers
- `'sgd'` - Stochastic Gradient Descent
- `'adam'` - Adam optimizer
- `'rmsprop'` - RMSprop
- `'adagrad'` - AdaGrad
- `'adamw'` - AdamW (Adam with weight decay)

## Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run specific test files
python -m pytest tests/test_neural_network.py
python -m pytest tests/test_layers.py -v
```

## Project Structure

```
deep-learning-framework/
├── deep_learning/           # Main package
│   ├── __init__.py
│   ├── neural_network.py    # Main NeuralNetwork class
│   ├── layers.py           # Layer implementations
│   ├── activation.py       # Activation functions
│   ├── optimizers.py       # Optimization algorithms
│   └── loss.py            # Loss functions
├── tests/                  # Test suite
│   ├── test_neural_network.py
│   ├── test_layers.py
│   ├── test_activation.py
│   ├── test_optimizers.py
│   └── test_loss.py
├── examples/               # Example scripts
│   ├── classification_demo_simple.py
|   ├── classification_demo.py
│   └── regression_demo.py
├── setup.py               # Package setup
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`python -m pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Performance Notes

This framework is built for educational purposes and understanding. For production use, consider:
- Using GPU acceleration (e.g., with CuPy)
- Implementing more efficient matrix operations
- Adding support for convolutional and recurrent layers
- Using established frameworks like TensorFlow or PyTorch for large-scale applications

## License
All Rights Resvered By Author Aric Hurkman. All Files herein are fo the purpose to showcase the ablities of the Author, and not for any other use without pramission by the Author 

## Acknowledgments

- Built as an educational tool to understand deep learning fundamentals
- Inspired by popular frameworks like Keras and PyTorch
- Thanks to the NumPy and Matplotlib communities

## Contact

**Aric Hurkman**  
Email: arichurkman@gmail.com  
GitHub: https://github.com/aturs3001

---

*This framework demonstrates core deep learning concepts and provides a foundation for understanding how neural networks work under the hood.*