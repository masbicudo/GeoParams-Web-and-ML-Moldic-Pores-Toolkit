import torch
import torch.nn as nn

class BetaVAEEncoder(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64→32
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32→16
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16→8
        )
        self.fc_mu = nn.Linear(8*8*32, latent_dim)
        self.fc_logvar = nn.Linear(8*8*32, latent_dim)
    
    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class BetaVAEDecoder(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.fc_dec = nn.Linear(latent_dim, 8*8*32)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),  # 8→16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),   # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 2, stride=2),    # 32→64
            nn.Sigmoid(),  # Output in range [0, 1]
        )
    
    def forward(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 32, 8, 8)
        x_recon = self.deconv(h)
        return x_recon

class BetaVAE(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.encoder = BetaVAEEncoder(latent_dim)
        self.decoder = BetaVAEDecoder(latent_dim)
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def beta_vae_loss_function(x, x_recon, mu, logvar, beta=4.0):
    # Reconstruction loss (MSE for binary masks)
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence between latent z and N(0,1)
    # Encourages disentangled latent space
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld_loss

def beta_vae_loss_function_a(outputs, target_or_inputs, beta=4.0):
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise ValueError("Outputs must be a tuple or list of (x_recon, mu, logvar)")
    x_recon, mu, logvar = outputs
    x = target_or_inputs
    if isinstance(x, (tuple, list)) and len(x) == 3:
        x = x[1] # Assume x is (input, target), take target
    return beta_vae_loss_function(x, x_recon, mu, logvar, beta=beta)
