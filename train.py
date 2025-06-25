import torch
import argparse
import os
import logging
import sys
from datetime import datetime
from tqdm import tqdm
from model import VQVAE, UNet3D
from SDFDataloader import CreateDataLoader
from diffusion import Diffusion, linear_beta_schedule
from torch.utils.tensorboard import SummaryWriter


# 设置日志记录器，同时输出到终端和文件
def setup_logger(log_path):
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # 创建记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理程序
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件处理程序
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 创建终端处理程序
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train VQVAE and Diffusion for SDF')
    parser.add_argument('--dataroot', type=str, default='./data', help='root directory of the dataset')
    parser.add_argument('--dataset_mode', type=str, default='snet', help='dataset type')
    parser.add_argument('--cat', type=str, default='chair', help='ShapeNet cat, use "all" for all categories')
    parser.add_argument('--res', type=int, default=64, help='SDF resolution')
    parser.add_argument('--trunc_thres', type=float, default=0.1, help='SDF truncation threshold')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--max_dataset_size', type=int, default=6778, help='size of dataset')
    parser.add_argument('--distributed', action='store_true', help='use distributed training')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    
    # VQVAE parameters
    parser.add_argument('--num_embeddings', type=int, default=512, help='encoder embedding size')
    parser.add_argument('--embedding_dim', type=int, default=64, help='encoder embedding dimension')
    parser.add_argument('--beta', type=float, default=0.25, help='VQ commitment loss weight')
    
    # training settings
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--save_every', type=int, default=10, help='save model every N epochs')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='model output directory')
    
    # diffusion parameters
    parser.add_argument('--phase', type=str, default='vqvae', choices=['vqvae', 'diffusion'], 
                        help='vqvqe or diffusion phase')
    parser.add_argument('--vqvae_checkpoint', type=str, default='./checkpoints/vqvqe/vqvae_best.pt', 
                        help='path of pretrained VQVAE model')
    parser.add_argument('--timesteps', type=int, default=1000, help='timesteps for diffusion model')
    parser.add_argument('--diffusion_lr', type=float, default=2e-5, help='learning rate for diffusion model')
    parser.add_argument('--diffusion_epochs', type=int, default=200, help='training epochs for diffusion model')
    
    parser.add_argument('--logdir', type=str, default='./logs', help='TensorBoard log directory')
    
    parser.add_argument('--debug', action='store_true', help='启用调试输出')
    
    return parser.parse_args()

def train_vqvae(train_dl, test_dl, opt, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"device: {device}")
    
    # initialize VQVAE model
    model = VQVAE(num_embeddings=opt.num_embeddings, embedding_dim=opt.embedding_dim, beta=opt.beta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    vqvae_output_dir = os.path.join(opt.output_dir, 'vqvae')
    os.makedirs(vqvae_output_dir, exist_ok=True)

    # 初始化TensorBoard writer
    log_dir = os.path.join(opt.logdir, 'vqvae_logs')
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logs will be saved to: {log_dir}")
    
    # training
    best_loss = float('inf')
    global_step = 0

    for epoch in range(opt.epochs):
        model.train()
        epoch_loss = 0
        recon_epoch_loss = 0
        vq_epoch_loss = 0
        
        with tqdm(total=len(train_dl), desc=f"Epoch {epoch+1}/{opt.epochs}") as pbar:
            for i, batch in enumerate(train_dl):
                sdf = batch['sdf'].to(device)
                
                # forward
                x_recon, loss_vq = model(sdf)
                
                # reconstruction loss
                recon_loss = torch.nn.MSELoss()(x_recon, sdf)
                
                # total loss
                loss = recon_loss + loss_vq
                
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                recon_epoch_loss += recon_loss.item()
                vq_epoch_loss += loss_vq.item()

                if global_step % 10 == 0:  # 每10个step记录一次
                    writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                    writer.add_scalar('Train/BatchReconLoss', recon_loss.item(), global_step)
                    writer.add_scalar('Train/BatchVQLoss', loss_vq.item(), global_step)
                global_step += 1

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix(loss=loss.item(), recon=recon_loss.item(), vq=loss_vq.item())
        
        avg_loss = epoch_loss / len(train_dl)
        avg_recon_loss = recon_epoch_loss / len(train_dl)
        avg_vq_loss = vq_epoch_loss / len(train_dl)
        
        # 记录epoch级别的训练loss
        writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
        writer.add_scalar('Train/EpochReconLoss', avg_recon_loss, epoch)
        writer.add_scalar('Train/EpochVQLoss', avg_vq_loss, epoch)
        #writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch)

        logger.info(f"Epoch {epoch+1}: total loss={avg_loss:.6f}, reconstruct loss={avg_recon_loss:.6f}, VQ loss={avg_vq_loss:.6f}")
        
        # save model
        if (epoch + 1) % opt.save_every == 0 or epoch == opt.epochs - 1:
            save_path = os.path.join(opt.output_dir, f'vqvae_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            logger.info(f"model saved to {save_path}")
        
        # eval
        if test_dl is not None:
            model.eval()
            val_loss = 0
            val_recon_loss = 0
            val_vq_loss = 0
            
            with torch.no_grad():
                for batch in test_dl:
                    sdf = batch['sdf'].to(device)
                    x_recon, loss_vq = model(sdf)
                    recon_loss = torch.nn.MSELoss()(x_recon, sdf)
                    loss = recon_loss + loss_vq
                    
                    val_loss += loss.item()
                    val_recon_loss += recon_loss.item()
                    val_vq_loss += loss_vq.item()
            
            val_loss /= len(test_dl)
            val_recon_loss /= len(test_dl)
            val_vq_loss /= len(test_dl)

            # 记录验证loss到TensorBoard
            writer.add_scalar('Val/EpochLoss', val_loss, epoch)
            writer.add_scalar('Val/EpochReconLoss', val_recon_loss, epoch)
            writer.add_scalar('Val/EpochVQLoss', val_vq_loss, epoch)
            
            # 记录训练和验证loss的对比
            writer.add_scalars('Loss/Comparison', {
                'Train': avg_loss,
                'Val': val_loss
            }, epoch)
            
            logger.info(f"eval: total loss={val_loss:.6f}, reconstruct loss={val_recon_loss:.6f}, VQ loss={val_vq_loss:.6f}")
            
            # save best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_path = os.path.join(opt.output_dir, 'vqvae_best.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, best_path)
                logger.info(f"best model saved to {best_path}")

    writer.close()
    logger.info("training complete!")
    logger.info(f"View training logs with: tensorboard --logdir {log_dir}")
    return model

def load_pretrained_vqvae(checkpoint_path, device, logger):
    """load pretrained VQVAE model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = VQVAE(num_embeddings=512, embedding_dim=64, beta=0.25).to(device)
    model.load_state_dict(checkpoint['model_state_dict']) # load weights
    model.eval()
    logger.info(f"Pretrained VQVAE model loaded, checkpoint epoch: {checkpoint['epoch']}")
    return model

def train_diffusion(train_dl, test_dl, opt, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"device: {device}")
    
    vqvae = load_pretrained_vqvae(opt.vqvae_checkpoint, device, logger)
    for param in vqvae.parameters():
        param.requires_grad = False
    
    # 创建UNet3D模型作为扩散模型的骨干网络
    latent_dim = 64
    unet_model = UNet3D(input_dim=latent_dim).to(device)
    
    # 先将模型移至设备，然后再初始化扩散模型
    diffusion_model = Diffusion(timesteps=opt.timesteps, model=unet_model).to(device)
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=opt.diffusion_lr)
    diffusion_output_dir = os.path.join(opt.output_dir, 'diffusion')
    os.makedirs(diffusion_output_dir, exist_ok=True)
    
    # 初始化TensorBoard writer
    log_dir = os.path.join(opt.logdir, 'diffusion_logs')
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logs will be saved to: {log_dir}")
    
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(opt.diffusion_epochs):
        unet_model.train()
        epoch_loss = 0.0
        
        with tqdm(total=len(train_dl), desc=f"Epoch {epoch+1}/{opt.diffusion_epochs}") as pbar:
            for i, batch in enumerate(train_dl):
                sdf = batch['sdf'].to(device)
                
                with torch.no_grad():
                    z_e = vqvae.encoder(sdf)
                    z_q, _, _ = vqvae.quantizer(z_e)
                
                # 随机选择时间步
                t = torch.randint(0, opt.timesteps, (z_q.shape[0],), device=device)
                
                # 计算扩散模型损失
                loss = diffusion_model.p_losses(z_q, t)
                
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # 记录每个batch的loss
                if global_step % 10 == 0:  # 每10个step记录一次
                    writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                
                global_step += 1
                
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(train_dl)
        
        # 记录每个epoch的平均loss
        writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
        
        logger.info(f"Epoch {epoch+1}: avg loss = {avg_loss:.6f}")
        
        if (epoch + 1) % opt.save_every == 0 or epoch == opt.diffusion_epochs - 1:
            save_path = os.path.join(diffusion_output_dir, f'diffusion_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': unet_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'diffusion_params': {
                    'timesteps': opt.timesteps,
                    'betas': diffusion_model.betas.cpu().numpy().tolist()
                },
                'loss': avg_loss,
            }, save_path)
            logger.info(f"diffusion model saved to {save_path}")
        
        # eval
        if test_dl is not None:
            unet_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in test_dl:
                    sdf = batch['sdf'].to(device)
                    
                    # 编码
                    z_e = vqvae.encoder(sdf)
                    z_q, _, _ = vqvae.quantizer(z_e)
                    
                    # 随机时间步
                    t = torch.randint(0, opt.timesteps, (z_q.shape[0],), device=device)
                    
                    # 计算损失
                    loss = diffusion_model.p_losses(z_q, t)
                    val_loss += loss.item()
            
            val_loss /= len(test_dl)
            
            # 记录验证loss到TensorBoard
            writer.add_scalar('Val/EpochLoss', val_loss, epoch)
            
            # 记录训练和验证loss的对比
            writer.add_scalars('Loss/Comparison', {
                'Train': avg_loss,
                'Val': val_loss
            }, epoch)
            
            logger.info(f"Eval: avg loss = {val_loss:.6f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_path = os.path.join(diffusion_output_dir, 'diffusion_best.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': unet_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'diffusion_params': {
                        'timesteps': opt.timesteps,
                        'betas': diffusion_model.betas.cpu().numpy().tolist()
                    },
                    'loss': best_loss,
                }, best_path)
                logger.info(f"Best diffusion model saved to {best_path}")
    
    writer.close()
    logger.info("Diffusion training complete!")
    logger.info(f"View training logs with: tensorboard --logdir {log_dir}")
    return unet_model

def main():
    opt = parse_args()
    
    # 设置日志文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if opt.phase == 'vqvae':
        log_path = os.path.join(opt.logdir, 'vqvae_logs', f'training_{timestamp}.txt')
    else:  # diffusion
        log_path = os.path.join(opt.logdir, 'diffusion_logs', f'training_{timestamp}.txt')
    
    # 设置日志记录器
    logger = setup_logger(log_path)
    
    if opt.debug:
        torch.autograd.set_detect_anomaly(True)
        logger.info("调试模式已启用")
    
    logger.info(f"Start training，logs will be saved to {log_path}")
    logger.info(f"parameters: {vars(opt)}")
    
    train_dl, test_dl, test_dl_for_eval = CreateDataLoader(opt)
    logger.info(f"Train Set Size: {len(train_dl)}, Test set Size: {len(test_dl)}")
    
    if opt.phase == 'vqvae':
        logger.info("Training VQVAE model...")
        model = train_vqvae(train_dl, test_dl, opt, logger)

    elif opt.phase == 'diffusion':
        logger.info("Training Diffusion model...")
        model = train_diffusion(train_dl, test_dl, opt, logger)
        
    else:
        error_msg = f"unknown training: {opt.phase}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return model

if __name__ == "__main__":
    main()