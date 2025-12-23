import torch
import torch.nn as nn

class StyleAEEncoder(nn.Module):
    def __init__(self, latent_dim=8, size=256, kernel=3, levels=3):
        super().__init__()
        seq = []
        channels = [3] + [2**(3+it) for it in range(levels)]
        for it in range(levels):
            seq.append(nn.Conv2d(channels[it], channels[it+1], kernel, padding=1))
            seq.append(nn.ReLU())
            seq.append(nn.MaxPool2d(2))  # Downsample by 2
        self.conv = nn.Sequential(*seq)
        sz = size // (2**levels)
        self.fc_mu = nn.Linear(sz*sz*channels[-1], latent_dim)
        self.fc_logvar = nn.Linear(sz*sz*channels[-1], latent_dim)

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

class StyleAEDecoder(nn.Module):
    def __init__(self, latent_dim=8, size=256, levels=3):
        super().__init__()
        sz = size // (2**levels)
        channels = [3] + [2**(3+it) for it in range(levels)]
        self.fc_dec = nn.Linear(latent_dim, sz*sz*channels[-1])
        seq = []
        for it in range(levels):
            seq.append(nn.ConvTranspose2d(channels[levels - it], channels[levels - it - 1], 2, stride=2))
            seq.append(nn.ReLU())
        self.deconv = nn.Sequential(
            *seq[:-1],  # All but last ReLU
            nn.Sigmoid(),  # Output in range [0, 1]
        )
        self.size = size
        self.levels = levels

    def forward(self, z):
        sz = self.size // (2**self.levels)
        h = self.fc_dec(z)
        h = h.view(-1, 2**(2+self.levels), sz, sz)
        x_recon = self.deconv(h)
        return x_recon

class StyleAE(nn.Module):
    def __init__(self, latent_dim=8, size=256, levels=3):
        super().__init__()
        self.encoder = StyleAEEncoder(latent_dim, size, levels=levels)
        self.decoder = StyleAEDecoder(latent_dim, size, levels=levels)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def style_ae_loss_function(x, x_recon, mu, logvar, beta=4.0):
    # Reconstruction loss (MSE for binary masks)
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence between latent z and N(0,1)
    # Encourages disentangled latent space
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld_loss
