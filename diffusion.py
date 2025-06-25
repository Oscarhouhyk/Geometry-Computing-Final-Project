import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# This code defines a simple diffusion model for 3D data.
def linear_beta_schedule(timesteps):
    beta_start, beta_end = 1e-4, 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

class Diffusion(nn.Module):
    def __init__(self, timesteps=1000, model=None):
        super().__init__()
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.model = model

    def q_sample(self, x0, t, noise):
        sqrt_ab = self.alpha_bars[t] ** 0.5
        sqrt_mab = (1 - self.alpha_bars[t]) ** 0.5
        return sqrt_ab[:, None, None, None, None] * x0 + sqrt_mab[:, None, None, None, None] * noise

    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_noisy)  # 3D U-Net输出预测噪声
        loss = nn.MSELoss()(noise_pred, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x, t):
        # 演示: 简单反向采样
        noise_pred = self.model(x)
        alpha_t = self.alpha_bars[t]
        beta_t = self.betas[t]
        mean = (1. / torch.sqrt(self.alphas[t])) * \
            (x - ((1 - self.alphas[t]) / torch.sqrt(1 - alpha_t)) * noise_pred)
        if t > 0:
            z = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x = mean + sigma_t * z
        else:
            x = mean
        return x
