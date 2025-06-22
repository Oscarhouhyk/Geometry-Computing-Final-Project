import torch
import torch.nn as nn
import torch.nn.functional as F
class LatentEncoder(nn.Module):
    pass
class LatentDecoder(nn.Module):
    pass
class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.encoder3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.decoder1 = nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1)
        self.decoder2 = nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1)
        self.decoder3 = nn.ConvTranspose3d(32, 1, kernel_size=3, padding=1)
    def forward(self, x):#predict the noise for the diffusion model
        x1= F.relu(self.encoder1(x))
        x2 = F.relu(self.encoder2(x1))
        x3 = F.relu(self.encoder3(x2))
        x4 = F.relu(self.decoder1(x3))
        x5 = F.relu(self.decoder2(x4))
        x6 = self.decoder3(x5)
        return x6
#print("torch version:", torch.__version__)