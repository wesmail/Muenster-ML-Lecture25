# Lecture-7: Graph Neural Networks

In this lecture, we will explore the foundations of **Graph Neural Networks (GNNs)**. We will begin by implementing a basic graph learning rule of the form:

$$
h = \sigma(WAX)
$$

where:
- $X$ is the input node feature matrix,
- $A$ is the adjacency matrix (possibly normalized),
- $W$ is a learnable weight matrix,
- $\sigma$ is a non-linear activation function (e.g., ReLU),
- $h$ is the output node representation.

After understanding this basic propagation rule, we will discuss two important GNN architectures:

- **Kipf & Welling's Graph Convolutional Network (GCN)**, which simplifies spectral graph convolutions.
- **Graph Attention Network (GAT)**, which introduces attention mechanisms to learn the importance of neighboring nodes.