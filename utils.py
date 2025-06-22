import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from skimage import measure
import numpy as np
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