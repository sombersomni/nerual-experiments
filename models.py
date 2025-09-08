import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=64, input_size=32, num_channels=3):
        super(VAEDecoder, self).__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        
        # Calculate the size after first FC layer
        # We'll start with 4x4 feature maps
        self.fc_size = 4
        self.fc_channels = 256
        
        # First FC layer to reshape latent vector
        self.fc = nn.Linear(latent_dim, self.fc_size * self.fc_size * self.fc_channels)
        
        # Transpose convolution layers to upsample
        self.decoder = nn.Sequential(
            # 4x4x256 -> 8x8x128
            nn.ConvTranspose2d(self.fc_channels, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 8x8x128 -> 16x16x64  
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 16x16x64 -> 32x32x32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 32x32x32 -> 32x32x3 (final output)
            nn.ConvTranspose2d(32, num_channels, 3, stride=1, padding=1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def forward(self, z):
        # Reshape latent vector to feature map
        x = self.fc(z)
        x = x.view(-1, self.fc_channels, self.fc_size, self.fc_size)
        
        # Decode to image
        x = self.decoder(x)
        return x


class MultiTaskImageTransformer(nn.Module):
    def __init__(self, input_size=28, num_channels=1, patch_size=4, embed_dim=128, 
                 num_heads=8, num_layers=6, num_classes=10, latent_dim=64):
        super(MultiTaskImageTransformer, self).__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_patches = (input_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Linear(patch_size * patch_size * num_channels, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, self.num_patches + 1)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # VAE latent space heads
        self.mu_head = nn.Linear(embed_dim, latent_dim)
        self.logvar_head = nn.Linear(embed_dim, latent_dim)
        
        # Embedding head (for triplet loss) - operates on latent samples
        self.embedding_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128)
        )
        
        # Classification head - operates on embeddings
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # VAE Decoder for reconstruction
        self.decoder = VAEDecoder(latent_dim, input_size, num_channels)
        
    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
            
    def forward(self, x, reconstruct=True):
        batch_size = x.size(0)
        original_x = x.clone()  # Keep original for reconstruction loss
        
        # Convert image to patches
        x = x.view(batch_size, self.num_channels, self.input_size, self.input_size)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.patch_size * self.patch_size * self.num_channels)
        
        # Patch embedding
        x = self.patch_embed(patches)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer
        x = self.transformer(x)
        
        # Use class token for latent space
        cls_output = x[:, 0]  # First token is class token
        
        # VAE latent space
        mu = self.mu_head(cls_output)
        logvar = self.logvar_head(cls_output)
        z = self.reparameterize(mu, logvar)
        
        # Get embeddings from latent sample
        embeddings = self.embedding_head(z)
        
        # Classification from embeddings
        logits = self.classifier(embeddings)
        
        outputs = {
            'embeddings': embeddings,
            'logits': logits,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
        
        # Optionally compute reconstruction
        if reconstruct:
            reconstruction = self.decoder(z)
            # Reshape original image to match decoder output
            original_reshaped = original_x.view(batch_size, self.num_channels, self.input_size, self.input_size)
            outputs['reconstruction'] = reconstruction
            outputs['original'] = original_reshaped
            
        return outputs


# Keep backward compatibility with old ImageTransformer
class ImageTransformer(MultiTaskImageTransformer):
    def __init__(self, input_size=28, num_channels=1, patch_size=4, embed_dim=128, num_heads=8, num_layers=6, num_classes=10):
        super().__init__(input_size, num_channels, patch_size, embed_dim, num_heads, num_layers, num_classes)
        
    def forward(self, x):
        outputs = super().forward(x, reconstruct=False)
        return outputs['embeddings'], outputs['logits']


class TripletLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()
        
    def forward(self, mu, logvar):
        """Compute KL divergence loss for VAE"""
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean()


class ReconstructionLoss(nn.Module):
    def __init__(self, loss_type='mse'):
        super(ReconstructionLoss, self).__init__()
        self.loss_type = loss_type
        
    def forward(self, reconstruction, original):
        """Compute reconstruction loss"""
        if self.loss_type == 'mse':
            return F.mse_loss(reconstruction, original, reduction='mean')
        elif self.loss_type == 'bce':
            return F.binary_cross_entropy(reconstruction, original, reduction='mean')
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")


class MultiTaskLoss(nn.Module):
    def __init__(self, triplet_weight=1.0, classification_weight=0.5, 
                 reconstruction_weight=1.0, kl_weight=0.01, margin=2.0):
        super(MultiTaskLoss, self).__init__()
        self.triplet_weight = triplet_weight
        self.classification_weight = classification_weight
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        
        self.triplet_loss = TripletLoss(margin=margin)
        self.classification_loss = nn.CrossEntropyLoss()
        self.reconstruction_loss = ReconstructionLoss('mse')
        self.kl_loss = KLDivergenceLoss()
        
    def forward(self, anchor_outputs, positive_outputs, negative_outputs, labels):
        """Compute multi-task loss"""
        losses = {}
        
        # Triplet loss
        triplet_l = self.triplet_loss(
            anchor_outputs['embeddings'],
            positive_outputs['embeddings'], 
            negative_outputs['embeddings']
        )
        losses['triplet'] = triplet_l
        
        # Classification loss
        classification_l = self.classification_loss(anchor_outputs['logits'], labels)
        losses['classification'] = classification_l
        
        # KL divergence loss
        kl_l = self.kl_loss(anchor_outputs['mu'], anchor_outputs['logvar'])
        losses['kl'] = kl_l
        
        # Reconstruction loss (only if reconstruction is available)
        if 'reconstruction' in anchor_outputs:
            reconstruction_l = self.reconstruction_loss(
                anchor_outputs['reconstruction'], 
                anchor_outputs['original']
            )
            losses['reconstruction'] = reconstruction_l
        else:
            reconstruction_l = torch.tensor(0.0, device=triplet_l.device)
            losses['reconstruction'] = reconstruction_l
            
        # Total loss
        total_loss = (self.triplet_weight * triplet_l + 
                     self.classification_weight * classification_l + 
                     self.reconstruction_weight * reconstruction_l + 
                     self.kl_weight * kl_l)
        
        losses['total'] = total_loss
        
        return total_loss, losses