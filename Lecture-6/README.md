# Lecture-6: Recurrent Neural Networks

In this lecture, we explore **sequence models** and their crucial role in modeling time-dependent or sequential data. The focus will be on:

- Recurrent Neural Networks (RNNs)
- Vanishing Gradient Problem
- Gated Recurrent Unit (GRU)
- Long Short-Term Memory (LSTM)

We'll build an intuitive understanding of how these models work, when to use them, and how to implement them in PyTorch.


## Notebooks

### 1. `Recurrent_Neural_Networks_Explained.ipynb`
This notebook explains the inner workings of RNNs and GRUs through clear, step-by-step PyTorch implementations using `nn.RNNCell` and `nn.GRUCell`.

### 2. `Gravitational_Waves_Detection.ipynb`
Uses **synthetic data** to simulate the detection of gravitational wave signals, showcasing how RNNs can be applied to real-world sequential signal detection problems.

### 3. `Track_Finding_PANDA.ipynb`
Applies RNNs to the **forward tracking system of the PANDA experiment**, demonstrating how to:
- Classify tracklets as **true** (from the same particle) or **false** (mixed),
- Predict missing hits from partial trajectories using many-to-many sequence modeling.

Happy learning!
