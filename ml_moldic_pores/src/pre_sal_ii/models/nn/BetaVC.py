import torch
import torch.nn as nn

class BetaVCEncoder(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(33, 24, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16→8
            
            nn.Conv2d(24, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8→4
        )
        self.fc_mu = nn.Linear(4*4*16, latent_dim)
        self.fc_logvar = nn.Linear(4*4*16, latent_dim)

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

class BetaVCDecoder(nn.Module):
    def __init__(self, latent_dim=8, use_sigmoid=False):
        super().__init__()
        self.fc_dec = nn.Linear(latent_dim, 4*4*16)
        seq = [
            nn.ConvTranspose2d(16, 8, 2, stride=2),   # 4→8
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 2, stride=2),    # 8→16
        ]
        if use_sigmoid:
            seq.append(nn.Sigmoid())  # Output in range [0, 1]
        self.deconv = nn.Sequential(*seq)
    
    def forward(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 16, 4, 4)
        x_recon = self.deconv(h)
        return x_recon

class BetaVC(nn.Module):
    def __init__(self, latent_dim=8, use_sigmoid=False):
        super().__init__()
        self.encoder = BetaVCEncoder(latent_dim)
        self.decoder = BetaVCDecoder(latent_dim, use_sigmoid=use_sigmoid)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def beta_VC_loss_function(x_recon_logits, target_mask, mu, logvar, beta=4.0):
    """
    Beta-VAE loss for binary segmentation (e.g., moldic pores).
    Args:
        x_recon_logits: raw outputs from decoder (no sigmoid)
        target_mask: binary target mask (0 or 1)
        mu, logvar: latent variables
        beta: weight for the KL term
    """
    
    # Reconstruction loss: predict moldic mask (binary)
    recon_loss = nn.functional.binary_cross_entropy_with_logits(x_recon_logits, target_mask, reduction='sum')

    # KL divergence regularization
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kld_loss

def beta_VC_loss_function_used_sigmoid(x_recon, target_mask, mu, logvar, beta=4.0):
    """
    Beta-VAE loss for binary segmentation (e.g., moldic pores).
    Args:
        x_recon: outputs from decoder
        target_mask: binary target mask (0 or 1)
        mu, logvar: latent variables
        beta: weight for the KL term
    """
    
    # Reconstruction loss: predict moldic mask (binary)
    recon_loss = nn.functional.binary_cross_entropy(x_recon, target_mask, reduction='sum')

    # KL divergence regularization
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kld_loss
