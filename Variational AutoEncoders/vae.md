# Variational Autoencoders: Theory and Implementation Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Theoretical Framework](#theoretical-framework)
3. [Core Components](#core-components)
4. [Implementation Details](#implementation-details)
5. [Applications](#applications)
6. [Advanced Topics](#advanced-topics)

## Introduction

Variational Autoencoders (VAEs) are powerful generative models that combine variational inference with neural networks. Unlike traditional autoencoders, VAEs learn a probabilistic mapping between the input space and a latent space, enabling both reconstruction and generation of new data.

## Theoretical Framework

### The ELBO and Variational Inference

The VAE optimizes the Evidence Lower BOund (ELBO):

```
ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
```

where:
- `p(x|z)` is the decoder (generative model)
- `q(z|x)` is the encoder (inference model)
- `p(z)` is the prior distribution (usually N(0,1))

This follows from Jensen's inequality:
```
log p(x) = log ∫ p(x,z)dz 
         = log ∫ p(x,z) q(z|x)/q(z|x) dz 
         ≥ E[log p(x|z)] - KL(q(z|x) || p(z))
```

### The Reparameterization Trick

To enable backpropagation through the sampling process, VAEs use the reparameterization trick:
```
z = μ + σ * ε, where ε ~ N(0,1)
```

This transforms the random sampling into a deterministic function of the parameters and a fixed noise source.

## Core Components

### 1. Encoder Network
```python
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
```

### 2. Decoder Network
```python
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
```

### 3. Loss Function
```python
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
```

## Implementation Details

### Full VAE Implementation
```python
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        # Decoder
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
```

### Training Loop
```python
def train_vae(model, dataloader, optimizer, epochs):
    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(dataloader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
```

## Applications

### 1. Generation
Generate new samples by sampling from the prior:
```python
def generate_samples(model, n_samples):
    with torch.no_grad():
        z = torch.randn(n_samples, model.latent_dim)
        samples = model.decoder(z)
    return samples
```

### 2. Reconstruction
Encode and decode input data:
```python
def reconstruct(model, data):
    with torch.no_grad():
        recon, _, _ = model(data)
    return recon
```

### 3. Latent Space Interpolation
```python
def interpolate(model, x1, x2, steps=10):
    z1 = model.encode(x1)[0]
    z2 = model.encode(x2)[0]
    alphas = torch.linspace(0, 1, steps)
    
    interpolations = []
    for alpha in alphas:
        z = z1 * (1-alpha) + z2 * alpha
        interpolations.append(model.decode(z))
    return interpolations
```

### 4. Disentanglement Analysis
```python
def analyze_disentanglement(model, data):
    # Compute MIG score
    encodings = model.encode(data)[0]
    mi_matrix = compute_mutual_information(encodings)
    mig_score = compute_mig(mi_matrix)
    
    # Latent traversal
    z = torch.zeros(1, model.latent_dim)
    traversals = []
    for dim in range(model.latent_dim):
        z_new = z.clone()
        z_new[0, dim] = torch.linspace(-3, 3, 10)
        traversals.append(model.decode(z_new))
    
    return mig_score, traversals
```

## Advanced Topics

### 1. β-VAE
Modify the loss function to control disentanglement:
```python
def beta_vae_loss(recon_x, x, mu, logvar, beta):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss
```

### 2. Conditional VAE
Add conditioning to both encoder and decoder:
```python
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = ConditionalEncoder(input_dim + condition_dim, hidden_dim, latent_dim)
        self.decoder = ConditionalDecoder(latent_dim + condition_dim, hidden_dim, output_dim)
```

### 3. Improved Probability Estimation
Using Annealed Importance Sampling:
```python
def estimate_log_likelihood(model, x, n_samples=1000):
    # Importance sampling estimate
    z_samples = model.sample_latent(n_samples)
    log_p_x_z = model.log_likelihood(x, z_samples)
    log_p_z = model.prior.log_prob(z_samples)
    log_q_z_x = model.encoder.log_prob(z_samples, x)
    
    # Log mean exp trick for numerical stability
    log_w = log_p_x_z + log_p_z - log_q_z_x
    log_p_x = torch.logsumexp(log_w, dim=0) - np.log(n_samples)
    
    return log_p_x
```

This implementation guide covers:
- Theoretical foundations of VAEs
- Core implementation details
- Key applications
- Advanced extensions and improvements

The code examples are designed to be modular and easily adaptable to different use cases. Each component can be modified or extended based on specific requirements.

For practical usage:
1. Start with the basic VAE implementation
2. Add applications based on your needs
3. Consider advanced extensions for specific requirements
4. Use the provided analysis tools to understand model behavior

