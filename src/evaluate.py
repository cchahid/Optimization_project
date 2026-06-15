import torch
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(model, test_loader):
    """
    Evaluates the PyTorch model on the provided test dataset.
    
    Args:
        model (nn.Module): The PyTorch model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        
    Returns:
        float: The final accuracy on the test set.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
    accuracy = correct / total
    return accuracy

def plot_convergence(sgd_losses, adam_losses, pso_losses, save_path='convergence.png'):
    """
    Plots the training convergence (loss over iterations/epochs) for SGD, Adam, and PSO.
    
    Args:
        sgd_losses (list or array): Loss history for SGD.
        adam_losses (list or array): Loss history for Adam.
        pso_losses (list or array): Loss history for PSO.
        save_path (str): The filename/path to save the plot. Defaults to 'convergence.png'.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(sgd_losses, label='SGD', color='blue', linestyle='--')
    plt.plot(adam_losses, label='Adam', color='green', linestyle='-.')
    plt.plot(pso_losses, label='PSO', color='red')
    
    plt.title('Optimizer Convergence Comparison')
    plt.xlabel('Iterations / Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path)
    print(f"Convergence plot saved to {save_path}")
    plt.close()

def generate_comparison_table(results_dict):
    """
    Generates and prints a comparison table from a dictionary of results.
    
    Args:
        results_dict (dict): A dictionary where keys are optimizer names (e.g., 'SGD')
                             and values are dictionaries with metrics (e.g., 'Execution Time', 'Final Loss', 'Test Accuracy').
    """
    df = pd.DataFrame.from_dict(results_dict, orient='index')
    print("\n--- Optimizer Comparison Results ---")
    print(df.to_string())
    print("------------------------------------")

