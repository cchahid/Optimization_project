# Optimization Trajectory Visualizer

## Overview
The **Optimization Trajectory Visualizer** is an interactive full-stack web application designed for the empirical analysis of mathematical optimization algorithms. It allows users to visualize, compare, and understand how different optimization algorithms navigate complex, non-convex loss surfaces to find the global minimum.

The project currently evaluates three distinct algorithms on the **2D Rastrigin function** (a popular performance test problem for optimization algorithms characterized by its highly multimodal landscape with many local minima).

## Supported Optimization Algorithms

1. **Stochastic Gradient Descent (SGD)**: A fundamental gradient-based optimizer that takes steps proportional to the negative of the gradient.
2. **Adam (Adaptive Moment Estimation)**: An advanced gradient-based optimizer that computes adaptive learning rates for each parameter by utilizing estimates of first and second moments of the gradients.
3. **Particle Swarm Optimization (PSO)**: A population-based, derivative-free optimization algorithm inspired by the social behavior of bird flocking or fish schooling.

## Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **Computation Engine**: PyTorch
- **Features**: 
  - Exposes an API endpoint (`/run-optimization`) to compute optimization trajectories dynamically.
  - Utilizes PyTorch's `requires_grad` and built-in optimizers (SGD, Adam) to perform autodiff and optimization steps.
  - Implements a custom PyTorch-based Particle Swarm Optimization (PSO) routine for vectorized and fast execution.

### Frontend
- **Framework**: Next.js (React) / TypeScript
- **Styling**: Tailwind CSS
- **Components**: `shadcn/ui` and `lucide-react` for a modern, sleek, "control panel" aesthetic.
- **Features**:
  - Interactive sidebar to tweak hyperparameters like **Learning Rate**, **Iterations**, and **Swarm Size** in real time.
  - Live visualization of the computed optimization trajectories.

## How to Run the Project

### Prerequisites
- Node.js & npm
- Python 3.8+ 

### Running the Backend
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Activate your virtual environment (if using one).
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```
   *The backend will run at `http://127.0.0.1:8000`.*

### Running the Frontend
1. Navigate to the `front` directory:
   ```bash
   cd ..
   cd front
   ```
2. Install the frontend dependencies:
   ```bash
   npm install
   ```
3. Start the Next.js development server:
   ```bash
   npm run dev
   ```
   *The frontend will run at `http://localhost:3000`.*

## How It Works
- The user configures the optimization hyperparameters on the frontend sidebar and clicks "Execute Optimization".
- The frontend sends a POST request to the FastAPI backend containing the `learning_rate`, `iterations`, and `swarm_size`.
- The backend initializes random starting positions and computes the step-by-step coordinates (paths) and loss values for SGD, Adam, and the PSO swarm using PyTorch over the Rastrigin landscape.
- The computed paths are sent back to the frontend, which renders the trajectories on the visualization canvas, allowing for an empirical comparison of their convergence behaviors and robustness against local minima.
