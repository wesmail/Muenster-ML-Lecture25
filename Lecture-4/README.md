## Lecture 4: Deep Model Building
In this lecture, we begin building **deep learning models** using PyTorch, introducing core concepts in neural network training and optimization.

This lecture introduces essential building blocks for deep learning:

- **Training Pipeline Overview**
  - Forward and backward passes
  - Gradient computation and parameter updates
- **PyTorch Implementation**
  - Model definition
  - Loss functions
  - Optimizers and training loops
- **Autograd & Computation Graphs**
  - Automatic differentiation with dynamic DAGs
- **Fully Connected Networks (MLPs)**
  - Structure: input, hidden, and output layers
  - Adding non-linearity through activation functions (ReLU, Tanh, LeakyReLU)
- **Optimization Techniques**
  - Challenges with vanilla gradient descent
  - Momentum-based optimization for stable and faster convergence
- **Loss Functions**
  - Binary cross-entropy and interpretation in classification problems

---

There are 2 accompanying **Jupyter notebooks**:

1. **Intro_to_PyTorch_Autograd_Activations_CrossEntropy.ipynb**  
   This notebook introduces:
   - Key PyTorch concepts including the `Autograd` system
   - Visualization and implementation of common activation functions
   - Manual implementation of the **binary cross-entropy loss**

2. **Binary_Classification_with_MLP.ipynb**  
   This notebook demonstrates:
   - Using scikit-learn’s `make_moons` to generate a toy binary classification dataset
   - Building and training a small "deep" neural network in PyTorch
   - Visualizing the **decision boundary** of the trained model

These notebooks are designed to help you connect theory with practical coding experience using intuitive toy examples.
