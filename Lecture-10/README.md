## Lecture 11: Machine Learning Applications in (Astro-) Particle Physics

In this lecture, we will explore three core topics at the intersection of machine learning and particle physics:

### 1. Anomaly Detection (Unsupervised)

- Learn how to detect rare or unexpected events **without labeled anomalies**.
- Understand how **Autoencoders** and **Variational Autoencoders (VAEs)** can reconstruct normal patterns and flag outliers.
- See how anomaly scores can be assigned using reconstruction error or likelihoods.
- Example applications include **data quality monitoring** and **new physics searches**.


### 2. Domain Adaptation

- Understand the concept of **domain shift**: when your training (simulation) and testing (real data) distributions differ.
- Learn about **Domain-Adversarial Neural Networks (DANN)**:
  - A model that simultaneously learns a classification task and tries to confuse a domain discriminator.
  - Encourages the network to learn **domain-invariant features**.
- Critical for transferring models from **simulations to real-world experiments**.


### 3. Uncertainty Quantification (UQ)

- Learn how to quantify **confidence in predictions** using techniques like **Monte Carlo Dropout**.
- Explore the difference between deterministic outputs and **Bayesian uncertainty estimates**.
- UQ is essential in **high-stakes decisions** like anomaly detection and scientific inference.
