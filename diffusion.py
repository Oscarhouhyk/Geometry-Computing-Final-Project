import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
'''# This code defines a simple diffusion model for 3D data.
def linear_beta_schedule(timesteps):
    beta_start, beta_end = 1e-4, 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

class Diffusion:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        sqrt_ab = self.alpha_bars[t] ** 0.5
        sqrt_mab = (1 - self.alpha_bars[t]) ** 0.5
        return sqrt_ab[:, None, None, None, None] * x0 + sqrt_mab[:, None, None, None, None] * noise
'''