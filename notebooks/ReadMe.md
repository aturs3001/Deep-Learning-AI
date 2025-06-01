# 📓 Jupyter Notebook Setup Guide

## 🚀 Quick Setup

### 1. **Install Jupyter in Your Environment**

```bash
# Make sure you're in your Deep Learning environment
# Windows:
"Deep Learning Env\Scripts\activate"

# Install Jupyter
pip install jupyter jupyterlab ipywidgets

# Optional: Install additional useful packages
pip install seaborn plotly
```

### 2. **Create Notebooks Directory**

```bash
# In your project root directory
mkdir notebooks
cd notebooks
```

### 3. **Copy the Notebook Files**

Copy these three notebook files to your `notebooks/` directory:
- `01_Getting_Started.ipynb`
- `02_Advanced_Examples.ipynb` 
- `03_Framework_Internals.ipynb`

### 4. **Start Jupyter**

```bash
# From the notebooks directory
jupyter notebook

# Or use JupyterLab (more modern interface)
jupyter lab
```

## 📚 Notebook Overview

### 🌟 **01_Getting_Started.ipynb**
**Perfect for beginners!**
- 🎯 Basic framework usage
- 📊 Simple binary classification
- 🏗️ Building your first neural network
- 📈 Training and evaluation
- 🎨 Basic visualization

**Time: ~20-30 minutes**

### 🔥 **02_Advanced_Examples.ipynb**  
**For intermediate users**
- 🌈 Multi-class classification
- 🟡 Non-linear data (circles)
- 📊 Regression problems
- ⚙️ Optimizer comparison
- 🧠 Advanced architectures with regularization
- 📈 Comprehensive performance analysis

**Time: ~45-60 minutes**

### 🔧 **03_Framework_Internals.ipynb**
**For advanced users and framework developers**
- 🧠 Individual layer operations
- 📊 Activation function deep dive
- ⚙️ Optimizer mechanics
- 🔄 Forward/backward propagation
- 🎛️ Weight initialization impact
- 📈 Network capacity analysis
- 🛡️ Regularization effects

**Time: ~60-90 minutes**

## 🎯 Learning Path

### 👶 **Beginner Path:**
1. Start with `01_Getting_Started.ipynb`
2. Run each cell step by step
3. Experiment with different parameters
4. Try modifying the network architecture

### 🧑‍🎓 **Intermediate Path:**
1. Complete Getting Started first
2. Move to `02_Advanced_Examples.ipynb`
3. Focus on the experiments that interest you most
4. Try the framework on your own datasets

### 🔬 **Advanced Path:**
1. Complete the first two notebooks
2. Dive into `03_Framework_Internals.ipynb`
3. Experiment with framework modifications
4. Implement custom components

## 🛠️ Troubleshooting

### **Common Issues:**

#### ❌ **Import Error: `ModuleNotFoundError: No module named 'deep_learning'`**
**Solution:**
```python
# Add this at the top of your notebook
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
```

#### ❌ **Kernel Dies or Crashes**
**Solution:**
- Restart the kernel: `Kernel → Restart`
- Check your environment has enough RAM
- Reduce batch sizes or model complexity

#### ❌ **Plots Not Showing**
**Solution:**
```python
# Add this magic command
%matplotlib inline

# Or use this for interactive plots
%matplotlib widget
```

#### ❌ **Slow Performance**
**Solution:**
- Reduce dataset sizes in examples
- Lower number of epochs
- Use smaller batch sizes
- Close other applications

## 🎨 Customization Tips

### **Make It Your Own:**

1. **Modify Datasets:**
   ```python
   # Try different dataset parameters
   X, y = make_classification(
       n_samples=1000,      # Change sample size
       n_features=4,        # Change feature count
       n_classes=3,         # Change number of classes
       random_state=42
   )
   ```

2. **Experiment with Architectures:**
   ```python
   # Try different layer sizes
   model.add_dense(units=64, activation='relu', input_size=4)
   model.add_dense(units=32, activation='relu')
   model.add_dense(units=16, activation='relu')
   ```

3. **Test Different Optimizers:**
   ```python
   # Compare optimizers
   for optimizer in ['sgd', 'adam', 'rmsprop']:
       model = NeuralNetwork(loss='mse', optimizer=optimizer)
       # ... train and compare
   ```

## 🚀 Next Steps

After completing the notebooks:

1. **Apply to Real Data:**
   - Load CSV files with pandas
   - Try image datasets
   - Work with time series data

2. **Extend the Framework:**
   - Add new activation functions
   - Implement new optimizers
   - Create custom loss functions

3. **Build Projects:**
   - House price prediction
   - Image classification
   - Text sentiment analysis

4. **Share Your Work:**
   - Export notebooks as HTML
   - Create presentations
   - Share on GitHub

## 🎉 Happy Learning!

The notebooks are designed to be:
- ✅ **Interactive** - Run and modify code
- ✅ **Educational** - Learn by doing
- ✅ **Progressive** - Build skills step by step
- ✅ **Practical** - Real examples and use cases

**Enjoy exploring your deep learning framework!** 🧠✨
