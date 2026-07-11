import math
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any proxy domain to fetch your backend routes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def rastrigin(x, y):
    """2D Rastrigin function for optimization landscape."""
    A = 10
    return 2 * A + (x**2 - A * torch.cos(2 * math.pi * x)) + (y**2 - A * torch.cos(2 * math.pi * y))

class OptimizationParams(BaseModel):
    learning_rate: float = 0.01
    iterations: int = 50
    num_particles: int = 20

@app.post("/run-optimization")
def run_optimization(params: OptimizationParams):
    iterations = params.iterations
    lr = params.learning_rate
    num_particles = params.num_particles

    # 1. Random starting point for Gradient Descent (x, y in [-5.12, 5.12])
    start_x = torch.rand(1).item() * 10.24 - 5.12
    start_y = torch.rand(1).item() * 10.24 - 5.12

    # ===============================
    # SGD
    # ===============================
    x_sgd = torch.tensor([start_x], requires_grad=True)
    y_sgd = torch.tensor([start_y], requires_grad=True)
    opt_sgd = torch.optim.SGD([x_sgd, y_sgd], lr=lr)
    
    sgd_path = []
    for _ in range(iterations):
        opt_sgd.zero_grad()
        loss = rastrigin(x_sgd, y_sgd)
        loss.backward()
        
        # Record before step to get current position
        sgd_path.append({"x": x_sgd.item(), "y": y_sgd.item(), "loss": loss.item()})
        opt_sgd.step()

    # ===============================
    # Adam
    # ===============================
    x_adam = torch.tensor([start_x], requires_grad=True)
    y_adam = torch.tensor([start_y], requires_grad=True)
    opt_adam = torch.optim.Adam([x_adam, y_adam], lr=lr)

    adam_path = []
    for _ in range(iterations):
        opt_adam.zero_grad()
        loss = rastrigin(x_adam, y_adam)
        loss.backward()
        
        adam_path.append({"x": x_adam.item(), "y": y_adam.item(), "loss": loss.item()})
        opt_adam.step()

    # ===============================
    # PSO
    # ===============================
    # Initialize particles
    particles_x = torch.rand(num_particles) * 10.24 - 5.12
    particles_y = torch.rand(num_particles) * 10.24 - 5.12
    vel_x = torch.randn(num_particles) * 0.1
    vel_y = torch.randn(num_particles) * 0.1

    pbest_x = particles_x.clone()
    pbest_y = particles_y.clone()
    pbest_loss = torch.full((num_particles,), float('inf'))

    gbest_loss = float('inf')
    gbest_x, gbest_y = None, None

    w, c1, c2 = 0.7, 1.5, 1.5
    
    # an array of arrays representing the path of each particle over time
    pso_path = [[] for _ in range(num_particles)] 

    for _ in range(iterations):
        for i in range(num_particles):
            # Evaluate fitness
            loss = rastrigin(particles_x[i:i+1], particles_y[i:i+1]).item()
            
            pso_path[i].append({
                "x": particles_x[i].item(), 
                "y": particles_y[i].item(), 
                "loss": loss
            })

            # Update personal best
            if loss < pbest_loss[i]:
                pbest_loss[i] = loss
                pbest_x[i] = particles_x[i].item()
                pbest_y[i] = particles_y[i].item()

                # Update global best
                if loss < gbest_loss:
                    gbest_loss = loss
                    gbest_x = particles_x[i].item()
                    gbest_y = particles_y[i].item()

        # Update velocities and positions
        r1_x, r2_x = torch.rand(num_particles), torch.rand(num_particles)
        r1_y, r2_y = torch.rand(num_particles), torch.rand(num_particles)

        vel_x = w * vel_x + c1 * r1_x * (pbest_x - particles_x) + c2 * r2_x * (gbest_x - particles_x)
        vel_y = w * vel_y + c1 * r1_y * (pbest_y - particles_y) + c2 * r2_y * (gbest_y - particles_y)

        particles_x = particles_x + vel_x
        particles_y = particles_y + vel_y

    return {
        "sgd_path": sgd_path,
        "adam_path": adam_path,
        "pso_path": pso_path
    }

from src.dataset import get_dataloaders
from src.models import SimpleMLP
from src.train import train_gradient_descent, compute_baseline_loss
from src.optimizers import ParticleSwarmOptimizer
import torch.nn as nn
import torch.optim as optim

@app.get("/training-convergence")
def training_convergence():
    epochs = 100
    train_loader, test_loader, num_features, num_classes = get_dataloaders(batch_size=32)
    criterion = nn.CrossEntropyLoss()

    # --- Baseline (no optimization) ---
    model_baseline = SimpleMLP(num_features, num_classes)
    baseline_loss = compute_baseline_loss(model_baseline, train_loader, criterion, epochs=epochs)

    # --- SGD ---
    model_sgd = SimpleMLP(num_features, num_classes)
    opt_sgd = optim.SGD(model_sgd.parameters(), lr=0.05)
    sgd_loss, _ = train_gradient_descent(model_sgd, train_loader, opt_sgd, criterion, epochs=epochs)

    # --- Adam ---
    model_adam = SimpleMLP(num_features, num_classes)
    opt_adam = optim.Adam(model_adam.parameters(), lr=0.01)
    adam_loss, _ = train_gradient_descent(model_adam, train_loader, opt_adam, criterion, epochs=epochs)

    # --- PSO ---
    model_pso = SimpleMLP(num_features, num_classes)
    pso = ParticleSwarmOptimizer(model_pso, num_particles=30)
    
    # Extract all training data for PSO since it operates on the whole dataset per iteration
    X_train_list, y_train_list = [], []
    for X_batch, y_batch in train_loader:
        X_train_list.append(X_batch)
        y_train_list.append(y_batch)
    X_train_full = torch.cat(X_train_list)
    y_train_full = torch.cat(y_train_list)

    pso_loss = pso.optimize(X_train_full, y_train_full, criterion, iterations=epochs)

    return {
        "epochs": list(range(1, epochs + 1)),
        "sgd_loss": sgd_loss,
        "adam_loss": adam_loss,
        "pso_loss": pso_loss,
        "baseline_loss": baseline_loss
    }

