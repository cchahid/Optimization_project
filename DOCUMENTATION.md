# Backend Architecture & Deep-Dive Documentation
**Data Engineering Master's Project**

This document serves as a comprehensive, rigorous guide to the Python backend of the project. It breaks down the architecture, the deep-learning training loop mechanics, and the custom implementation of Particle Swarm Optimization (PSO). It is designed to act as a study guide for academic defense.

---

## 1. Architecture Overview

At the core of this project, the backend serves as the computational engine for two distinct domains:
1. **Training a Multi-Layer Perceptron (MLP)** on the Iris dataset.
2. **Simulating Optimization Trajectories** on a 2D Rastrigin function.

### The Role of FastAPI
**FastAPI** acts as the high-performance bridge between the heavy computational layer (PyTorch) and the client interface (Next.js/React frontend). 
- It uses asynchronous request handling to serve ML computations without blocking.
- It exposes RESTful endpoints (`/run-optimization` and `/training-convergence`) that trigger PyTorch computations on-demand.
- It serializes PyTorch tensors and mathematical arrays into JSON payloads, allowing the React frontend to seamlessly plot the data using `react-plotly.js`.

---

## 2. Core Domain 1: Neural Network Training (Iris Dataset)

The training domain uses standard gradient-descent-based optimizers (SGD and Adam) compared against our custom PSO. 

### 2.1 The Standard PyTorch Training Loop
Below is the core of the gradient descent training loop from `src/train.py`. The code block is heavily annotated to explain the *how* and *why* behind each line.

```python
# Iterate over batches provided by the DataLoader
for batch_features, batch_labels in train_loader:
    
    # 1. Zero the Gradients
    # PyTorch natively accumulates gradients on subsequent backward passes.
    # We must explicitly flush the gradients from the previous batch to 0.
    optimizer.zero_grad()
    
    # 2. Forward Pass
    # We pass the input features through the MLP. Under the hood, this computes
    # the affine transformations (W*x + b) and applies non-linear activations.
    outputs = model(batch_features)
    
    # 3. Loss Calculation
    # We compute the discrepancy between our model's predictions and the true labels.
    # For classification, we use Cross-Entropy Loss, which applies a softmax
    # internally and calculates the negative log-likelihood.
    loss = criterion(outputs, batch_labels)
    
    # 4. Backward Pass (Backpropagation)
    # This triggers the automatic differentiation engine (Autograd).
    # It computes the partial derivative of the loss with respect to every
    # trainable parameter (weight and bias) in the model using the chain rule.
    loss.backward()
    
    # 5. Optimizer Step
    # The optimizer iterates through all model parameters and updates them
    # in the direction opposite to the calculated gradients to minimize the loss.
    optimizer.step()
```

### 2.2 PyTorch Mechanics at the Tensor Level

Understanding what happens mathematically during these three critical PyTorch functions is essential for defense:

#### `optimizer.zero_grad()`
- **Mathematical Meaning**: Sets $\nabla_{\theta} L = 0$ for all parameters $\theta$.
- **Tensor Action**: Iterates through the model's parameters (`model.parameters()`) and sets their `.grad` attribute to a tensor of zeros (or `None`). Without this, gradients from batch $t$ would be added to batch $t+1$, resulting in incorrect, bloated updates.

#### `loss.backward()`
- **Mathematical Meaning**: Computes the gradient of the scalar loss $L$ with respect to the weights vector $\theta$: $\frac{\partial L}{\partial \theta}$.
- **Tensor Action**: PyTorch traverses the **Computational Graph** backward from the `loss` node. Because every tensor operation in the forward pass was tracked by Autograd, PyTorch applies the **Chain Rule** recursively. The resulting gradients are deposited directly into the `.grad` attribute of each respective parameter tensor.

#### `optimizer.step()`
- **Mathematical Meaning**: Applies the update rule to the weights. For standard SGD, this is: 
  $\theta_{t+1} = \theta_t - \eta \nabla_{\theta} L$
  Where $\eta$ is the learning rate.
- **Tensor Action**: The optimizer reads the `.grad` attribute of each parameter and modifies the actual data tensor `.data` by subtracting the gradient multiplied by the learning rate. For Adam, this step is much more complex, calculating exponentially moving averages of both the gradients (first moment) and the squared gradients (second moment) to apply an adaptive learning rate per parameter.

---

## 3. Core Domain 2: Custom Particle Swarm Optimization (PSO)

Unlike gradient descent, PSO is a *meta-heuristic, derivative-free* optimization algorithm. It searches the loss landscape by simulating a "swarm" of particles that share information about the best locations they've found.

### 3.1 PSO Mathematical Breakdown

The core of PSO relies on updating two properties for each particle at every iteration: **Velocity ($v$)** and **Position ($x$)**.

**The Velocity Update Equation:**
$$ v_{t+1} = w \cdot v_t + c_1 \cdot r_1 \cdot (pbest - x_t) + c_2 \cdot r_2 \cdot (gbest - x_t) $$

Where:
- $w \cdot v_t$: **Inertia Component**. $w$ (inertia weight) prevents the particle from changing direction too abruptly, maintaining search momentum.
- $c_1 \cdot r_1 \cdot (pbest - x_t)$: **Cognitive Component**. The particle's tendency to return to the best position *it personally* has ever found ($pbest$). $c_1$ is the cognitive coefficient, and $r_1$ is a random vector $[0, 1]$.
- $c_2 \cdot r_2 \cdot (gbest - x_t)$: **Social Component**. The particle's tendency to move towards the best position found by the *entire swarm* ($gbest$). $c_2$ is the social coefficient, and $r_2$ is a random vector $[0, 1]$.

**The Position Update Equation:**
$$ x_{t+1} = x_t + v_{t+1} $$

### 3.2 The PSO Implementation Code

Below is the heavily annotated core execution loop from `src/optimizers.py` (`ParticleSwarmOptimizer` class) where these mathematics are realized in PyTorch tensors:

```python
for i in range(iterations):
    for p in range(self.num_particles):
        
        # 1. Inject the particle's 1D position vector directly into the model's weights
        self._set_weights(positions[p])
        
        # 2. Forward Pass (No Gradients needed!)
        # Because PSO is derivative-free, we wrap this in torch.no_grad()
        # to save memory and skip building the computational graph.
        with torch.no_grad():
            outputs = self.model(X)
            loss = criterion(outputs, y).item()
        
        # 3. Update Personal Best (Cognitive Memory)
        if loss < pbest_values[p]:
            pbest_values[p] = loss
            pbest_positions[p] = positions[p].clone()
            
            # 4. Update Global Best (Social Memory)
            # If this particle found a new best for the whole swarm, record it.
            if loss < gbest_value:
                gbest_value = loss
                gbest_position = positions[p].clone()
                
    # 5. Generate stochastic factors (r1, r2) for the iteration
    # These inject randomness into the search, preventing premature convergence.
    r1 = torch.rand(self.num_particles, dim)
    r2 = torch.rand(self.num_particles, dim)
    
    # 6. Apply the Velocity Update Equation mathematically (Vectorized)
    # w * velocities                  -> Inertia
    # c1 * r1 * (pbest - positions)   -> Cognitive component
    # c2 * r2 * (gbest - positions)   -> Social component
    velocities = (self.w * velocities + 
                  self.c1 * r1 * (pbest_positions - positions) + 
                  self.c2 * r2 * (gbest_position - positions))
    
    # 7. Apply the Position Update Equation mathematically (Vectorized)
    positions = positions + velocities
```

### Why Vectorization Matters Here
Notice that while the fitness evaluation (forward pass) is done via a loop over the particles, the velocity and position updates (lines 6 & 7) are executed in a single vectorized PyTorch tensor operation. This avoids slow Python loops, utilizing optimized C++ backends (and optionally CUDA) to calculate the complex mathematical updates for hundreds of particles instantly.
