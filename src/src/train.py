import torch
import torch.optim as optim
from torchdiffeq import odeint

def train_flow_matcher(model, vector_field, dataloader, epochs=250, target_gap=-1.0):
    optimizer = optim.Adam(vector_field.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_physics_loss = 0.0
        
        for batch in dataloader:
            optimizer.zero_grad()
            # Latent encoding of real data
            with torch.no_grad():
                _, z_real, _ = model(batch)
            
            # Initial Gaussian noise
            z0 = torch.randn_like(z_real)
            t = torch.tensor([0.0, 1.0])
            
            # Solve ODE flow trajectory
            trajectory = odeint(vector_field, z0, t, method='euler')
            z1 = trajectory[-1]
            
            # Physics Surrogate Evaluation on generated endpoint
            y_pred = model.surrogate(z1)
            
            # Loss calculations
            loss_physics = torch.mean((y_pred - target_gap) ** 2)
            loss_dist = torch.mean((z1.mean(0) - z_real.mean(0))**2) + torch.mean((z1.std(0) - z_real.std(0))**2)
            
            loss_total = 5.0 * loss_physics + loss_dist
            loss_total.backward()
            optimizer.step()
            
            total_loss += loss_total.item()
            total_physics_loss += loss_physics.item()
            
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch} | Flow Loss: {total_loss:.4f} (Physics Penalty: {total_physics_loss:.4f})")
