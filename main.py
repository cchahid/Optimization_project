import time
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_dataloaders
from src.models import SimpleMLP
from src.train import train_gradient_descent
from src.optimizers import ParticleSwarmOptimizer
from src.evaluate import evaluate_model, plot_convergence, generate_comparison_table

def main():
    # 2. Set a fixed random seed for reproducibility
    torch.manual_seed(42)

    # 3. Get dataloaders
    # Using batch_size=150 ensures the entire training set (120 samples) is loaded at once
    train_loader, test_loader, num_features, num_classes = get_dataloaders(batch_size=150)
    
    # Extract full training batch for PSO
    X_train_full, y_train_full = next(iter(train_loader))

    # 4. Define the loss function and epochs
    criterion = nn.CrossEntropyLoss()
    epochs = 100

    results_dict = {}

    # ==========================
    # 5. Run SGD
    # ==========================
    print("Training with SGD...")
    model_sgd = SimpleMLP(input_dim=num_features, output_dim=num_classes)
    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
    
    start_time = time.time()
    sgd_losses, _ = train_gradient_descent(model_sgd, train_loader, optimizer_sgd, criterion, epochs=epochs)
    sgd_time = time.time() - start_time
    
    sgd_test_acc = evaluate_model(model_sgd, test_loader)
    
    results_dict['SGD'] = {
        'Execution Time (s)': sgd_time,
        'Final Training Loss': sgd_losses[-1],
        'Test Accuracy': sgd_test_acc
    }

    # ==========================
    # 6. Run Adam
    # ==========================
    print("Training with Adam...")
    model_adam = SimpleMLP(input_dim=num_features, output_dim=num_classes)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.01)
    
    start_time = time.time()
    adam_losses, _ = train_gradient_descent(model_adam, train_loader, optimizer_adam, criterion, epochs=epochs)
    adam_time = time.time() - start_time
    
    adam_test_acc = evaluate_model(model_adam, test_loader)
    
    results_dict['Adam'] = {
        'Execution Time (s)': adam_time,
        'Final Training Loss': adam_losses[-1],
        'Test Accuracy': adam_test_acc
    }

    # ==========================
    # 7. Run PSO
    # ==========================
    print("Training with PSO...")
    model_pso = SimpleMLP(input_dim=num_features, output_dim=num_classes)
    optimizer_pso = ParticleSwarmOptimizer(model_pso, num_particles=30)
    
    start_time = time.time()
    pso_losses = optimizer_pso.optimize(X_train_full, y_train_full, criterion, iterations=epochs)
    pso_time = time.time() - start_time
    
    pso_test_acc = evaluate_model(model_pso, test_loader)
    
    results_dict['PSO'] = {
        'Execution Time (s)': pso_time,
        'Final Training Loss': pso_losses[-1],
        'Test Accuracy': pso_test_acc
    }

    # ==========================
    # 8-10. Results & Visualization
    # ==========================
    generate_comparison_table(results_dict)
    plot_convergence(sgd_losses, adam_losses, pso_losses, save_path='convergence.png')

if __name__ == '__main__':
    main()

