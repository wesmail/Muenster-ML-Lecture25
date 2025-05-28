# Lecture-7: Attention Is All You Need  

In this lecture, we'll explore and implement key ideas from the Transformer architecture, focusing on:

### Multi-Head Attention (MHA)
- Understand the motivation behind attention
- Implement scaled dot-product attention and multi-head attention **from scratch**
- Explore how attention lets models focus on the **most relevant information**

### Normalization Layers
- Understand the role of **Layer Normalization** vs **Batch Normalization**
- Visualize and compare their behaviors in different contexts

### GPT Architecture
- Learn the building blocks of the **Generative Pretrained Transformer (GPT)**
- Build a mini-GPT model layer-by-layer using:
  - Multi-head attention
  - LayerNorm + residuals
  - Feed-forward MLPs

### Why Attention Matters

To make it intuitive, we'll start with a **simple task**:
> Given a bag of digits, predict the **largest digit**  
> (e.g., from three MNIST digits)

This example helps demonstrate:
- Why some inputs are **more important than others**
- How attention allows the model to **weigh information dynamically**

### What You'll Implement

- Positional encoding (sinusoidal)
- Scaled dot-product attention
- Multi-head attention
- Attention-based digit classifier
- GPT-style Transformer block
- Final GPT model using PyTorch