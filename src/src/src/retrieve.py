import torch

def retrieve_nearest_neighbor(z_ideal, decoder, X_real_tensor, smiles_list):
    """Projects ideal latent coordinate back to a 100% valid QM9 molecule via Manhattan distance."""
    with torch.no_grad():
        x_pred_prob = decoder(z_ideal)
        
    # Compute Manhattan distance against real dataset feature matrix
    distances = torch.sum(torch.abs(X_real_tensor - x_pred_prob), dim=1)
    best_idx = torch.argmin(distances).item()
    
    return smiles_list[best_idx], distances[best_idx].item()
