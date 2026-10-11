import torch
import torch.nn as nn

class MLPAutoencoder(nn.Module):
    """Joint Autoencoder and Physics Surrogate for 128-bit Morgan Fingerprints."""
    def __init__(self, input_dim=128, latent_dim=16):
        super(MLPAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )
        self.surrogate = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        y_pred = self.surrogate(z)
        return x_recon, z, y_pred


class VectorField(nn.Module):
    """Continuous Normalizing Flow ODE Vector Field Network."""
    def __init__(self, latent_dim=16):
        super(VectorField, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, t, z):
        t_vector = t.repeat(z.shape[0], 1) if t.dim() == 0 else t
        if t_vector.dim() == 1:
            t_vector = t_vector.unsqueeze(1)
        tz_input = torch.cat([z, t_vector], dim=1)
        return self.net(tz_input)
