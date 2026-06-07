import torch

def train_gradient_descent(model, train_loader, optimizer, criterion, epochs=50):
    """
    Standard PyTorch training loop using gradient descent-based optimizers.
    
    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): The optimizer (e.g., SGD, Adam).
        criterion (nn.Module): The loss function (e.g., CrossEntropyLoss).
        epochs (int): Number of training epochs. Defaults to 50.
        
    Returns:
        tuple: (epoch_losses, epoch_accuracies) lists containing the training metrics.
    """
    # 1. Set the model to training mode
    model.train()
    
    epoch_losses = []
    epoch_accuracies = []
    
    # 2. Loop through the specified number of epochs
    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # 3. Iterate over the train_loader
        for batch_features, batch_labels in train_loader:
            # 4. Standard PyTorch steps
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_features)
            
            # Calculate loss
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            # 5. Track loss and accuracy
            running_loss += loss.item() * batch_features.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
            
        # Calculate the average loss and accuracy for the epoch
        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
        
    # 6. Return the metrics
    return epoch_losses, epoch_accuracies

