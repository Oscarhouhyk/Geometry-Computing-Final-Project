import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from skimage import measure
import numpy as np
from torch import distributed as dist


#Fourier encoding
device = torch.device('cuda:0')
def extract_mesh(model, resolution=128, bounding_box=1.2, threshold=0.0, filename='output.obj'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    x = np.linspace(-bounding_box, bounding_box, resolution)
    grid = np.stack(np.meshgrid(x, x, x), axis=-1).reshape(-1, 3)  # shape: [N, 3]
    
    sdf_values = []
    batch_size = 32768
    with torch.no_grad():
        for i in range(0, len(grid), batch_size):
            pts = torch.tensor(grid[i:i+batch_size], dtype=torch.float32).to(device)
            sdf = model(pts).cpu().numpy()  # shape: [B, 1]
            sdf_values.append(sdf)

    sdf_all = np.concatenate(sdf_values, axis=0).reshape((resolution, resolution, resolution))
    
    verts, faces, normals, _ = measure.marching_cubes(sdf_all, level=threshold, spacing=(2*bounding_box/resolution,)*3)
   
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.export(filename)
    print(f"Mesh saved to {filename}")

def sdf_loss(model, coords, gt_sdf, gt_grad, lambda_weight=0.01):
    coords = coords.detach().clone().to(device).requires_grad_(True)  # 确保 coords 可以计算梯度

    pred_sdf = model(coords)
    loss_sdf = F.mse_loss(pred_sdf, gt_sdf)

    grad_sdf = torch.autograd.grad(
        outputs=pred_sdf,
        inputs=coords,
        grad_outputs=torch.ones_like(pred_sdf),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    loss_grad = F.mse_loss(grad_sdf, gt_grad)

    loss = loss_sdf + lambda_weight * loss_grad
    return loss


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    # NOTE: this class contains a bug regarding beta; see VectorQuantizer2 for
    # a fix and use legacy=False to apply that fix. VectorQuantizer2 can be
    # used wherever VectorQuantizer has been used before and is additionally
    # more efficient.
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, depth, height, width) for 3D data
        quantization pipeline:
            1. get encoder input (B,C,D,H,W)
            2. flatten input to (B*D*H*W,C)
        """
        # 检查输入维度
        if len(z.shape) == 5:  # 3D 数据 (B,C,D,H,W)
            # 重新排列维度为 (batch, depth, height, width, channel)
            z = z.permute(0, 2, 3, 4, 1).contiguous()
            z_flattened = z.view(-1, self.e_dim)
        else:  # 2D 数据 (B,C,H,W) - 原始代码逻辑
            z = z.permute(0, 2, 3, 1).contiguous()
            z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        if len(z.shape) == 5:  # 3D 数据
            z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
        else:  # 2D 数据
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel) or (batch, depth, height, width, channel)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            if len(shape) == 5:  # 3D 数据 (B,D,H,W,C)
                z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
            else:  # 2D 数据 (B,H,W,C)
                z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
    


def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()


def synchronize(local_rank=0):
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()
    # dist.barrier(device_ids=[local_rank])