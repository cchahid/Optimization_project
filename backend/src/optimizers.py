import torch

class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization (PSO) for PyTorch model weights.
    """
    def __init__(self, model, num_particles=30, c1=1.5, c2=1.5, w=0.7):
        self.model = model
        self.num_particles = num_particles
        self.c1 = c1
        self.c2 = c2
        self.w = w

    def _set_weights(self, weights_1d):
        """
        Helper function to inject a flat 1D tensor of weights back into 
        the PyTorch model's state dictionary parameters.
        """
        offset = 0
        with torch.no_grad():
            for param in self.model.parameters():
                numel = param.numel()
                # Extract the relevant slice and reshape it to the parameter's shape
                param_weights = weights_1d[offset:offset + numel].view_as(param)
                param.copy_(param_weights)
                offset += numel

    def optimize(self, X, y, criterion, iterations=50):
        """
        Executes the PSO algorithm to optimize the model's weights.
        """
        # 1. Extract the total number of parameters to determine search space dimension
        dim = sum(p.numel() for p in self.model.parameters())
        
        # 2. Initialize particle positions and velocities randomly
        # We use standard normal distribution for positions and smaller values for velocities
        positions = torch.randn(self.num_particles, dim)
        velocities = torch.randn(self.num_particles, dim) * 0.1
        
        # Track personal bests
        pbest_positions = positions.clone()
        pbest_values = torch.full((self.num_particles,), float("inf"))
        
        # Track global best
        gbest_position = None
        gbest_value = float("inf")
        
        loss_history = []
        
        # 4. Main loop
        for i in range(iterations):
            for p in range(self.num_particles):
                # Inject current particle position into the model
                self._set_weights(positions[p])
                
                # Evaluate fitness using standard PyTorch forward pass without gradients
                with torch.no_grad():
                    outputs = self.model(X)
                    loss = criterion(outputs, y).item()
                
                # 5. Update personal best (pbest)
                if loss < pbest_values[p]:
                    pbest_values[p] = loss
                    pbest_positions[p] = positions[p].clone()
                    
                    # Update global best (gbest)
                    if loss < gbest_value:
                        gbest_value = loss
                        gbest_position = positions[p].clone()
                        
            # 7. Track the global best loss at each iteration
            loss_history.append(gbest_value)
            
            # 6. Update velocities and positions using PSO formulas
            r1 = torch.rand(self.num_particles, dim)
            r2 = torch.rand(self.num_particles, dim)
            
            # Velocity update
            velocities = (self.w * velocities + 
                          self.c1 * r1 * (pbest_positions - positions) + 
                          self.c2 * r2 * (gbest_position - positions))
            
            # Position update
            positions = positions + velocities
            
        # 8. Inject the gbest position back into the model 
        if gbest_position is not None:
            self._set_weights(gbest_position)
            
        # 9. Return the list of global best losses
        return loss_history

