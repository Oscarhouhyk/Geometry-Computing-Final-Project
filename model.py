import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import VectorQuantizer
#from taming.modules.vqvae.quantize import VectorQuantizer
class LatentEncoder(nn.Module):#Assume input(B*64*64*64)->
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
        nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
        F.relu(),
        nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
        F.relu(),
        nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        )
    def forward(self, x):
        x = self.net(x)
        return x
    pass
class LatentDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
        nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1),
        F.relu(),
        nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1),
        F.relu(),
        nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1)
        )   
    def forward(self, x):
        x = self.net(x)
        return x
    pass
class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128):
        super().__init__()
        self.encoder = LatentEncoder()
        self.decoder = LatentDecoder()
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, loss_vq, _ = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, loss_vq
    pass
class UNet3D(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.encoder1 = nn.Conv3d(input_dim, 64, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.encoder3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.decoder1 = nn.ConvTranspose3d(256, 128, kernel_size=3, padding=1)
        self.decoder2 = nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1)
        self.decoder3 = nn.ConvTranspose3d(64, input_dim, kernel_size=3, padding=1)
    def forward(self, x):#predict the noise for the diffusion model
        x1= F.relu(self.encoder1(x))
        x2 = F.relu(self.encoder2(x1))
        x3 = F.relu(self.encoder3(x2))
        x4 = F.relu(self.decoder1(x3))
        x5 = F.relu(self.decoder2(x4))
        x6 = self.decoder3(x5)
        return x6
#print("torch version:", torch.__version__)