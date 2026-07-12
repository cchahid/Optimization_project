import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    """
    A simple Logistic Regression model using a single linear layer.
    """
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)


class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) with one hidden layer and ReLU activation.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=10):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

