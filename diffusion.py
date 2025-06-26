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
        # 获取模型的设备
        self.device = next(model.parameters()).device if model is not None else torch.device('cpu')
        
        # 确保所有张量都在同一设备上
        self.betas = linear_beta_schedule(timesteps).to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.model = model

    def q_sample(self, x0, t, noise):
        # 确保t在正确的设备上
        t = t.to(self.device)
        # 采样扩散过程中的噪声图像
        sqrt_ab = self.alpha_bars[t][:, None, None, None, None] ** 0.5
        sqrt_mab = (1 - self.alpha_bars[t][:, None, None, None, None]) ** 0.5
        return sqrt_ab * x0 + sqrt_mab * noise

    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_noisy, t)  # 3D U-Net输出预测噪声，加入时间步
        loss = nn.MSELoss()(noise_pred, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x, t):
        # 确保t是一个标量并且在正确的设备上
        t = torch.tensor([t], device=self.device)
        t_in = torch.full((x.shape[0],), t[0], device=self.device, dtype=torch.long)
        # 预测噪声
        noise_pred = self.model(x, t_in)  # 3D U-Net输出预测噪声，加入时间步
        
        # 计算去噪后的平均值
        alpha_t = self.alpha_bars[t]
        beta_t = self.betas[t]
        
        # 处理维度问题
        alpha_t = alpha_t[:, None, None, None, None]
        beta_t = beta_t[:, None, None, None, None]
        
        mean = (1. / torch.sqrt(self.alphas[t][:, None, None, None, None])) * \
            (x - ((1 - self.alphas[t][:, None, None, None, None]) / torch.sqrt(1 - alpha_t)) * noise_pred)
        
        if t.item() > 0:
            z = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x = mean + sigma_t * z
        else:
            x = mean
        
        return x
    
    def p_sample_loop(self, shape):
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t)
        return x