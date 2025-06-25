import torch
import argparse
import os
from tqdm import tqdm
from model import VQVAE
from SDFDataloader import CreateDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Train VQVAE for SDF')
    parser.add_argument('--dataroot', type=str, default='./data', help='root directory of the dataset')
    parser.add_argument('--dataset_mode', type=str, default='snet', help='dataset type')
    parser.add_argument('--cat', type=str, default='chair', help='ShapeNet cat, use "all" for all categories')
    parser.add_argument('--res', type=int, default=64, help='SDF resolution')
    parser.add_argument('--trunc_thres', type=float, default=0.1, help='SDF truncation threshold')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--max_dataset_size', type=int, default=6778, help='size of dataset')
    parser.add_argument('--distributed', action='store_true', help='是否使用分布式训练')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    # VQVAE参数
    parser.add_argument('--num_embeddings', type=int, default=512, help='encoder embedding size')
    parser.add_argument('--embedding_dim', type=int, default=64, help='encoder embedding dimension')
    parser.add_argument('--beta', type=float, default=0.25, help='VQ commitment loss weight')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--save_every', type=int, default=10, help='save model every N epochs')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='model output directory')
    
    # 调试参数
    parser.add_argument('--debug', action='store_true', help='启用调试输出')
    
    return parser.parse_args()

def train_vqvae(train_dl, test_dl, opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    
    # 初始化模型
    model = VQVAE(num_embeddings=opt.num_embeddings, embedding_dim=opt.embedding_dim, beta=opt.beta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    # 创建保存目录
    os.makedirs(opt.output_dir, exist_ok=True)
    
    # 开始训练
    best_loss = float('inf')
    for epoch in range(opt.epochs):
        model.train()
        epoch_loss = 0
        recon_epoch_loss = 0
        vq_epoch_loss = 0
        
        # 使用tqdm显示进度
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

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix(loss=loss.item(), recon=recon_loss.item(), vq=loss_vq.item())
        
        avg_loss = epoch_loss / len(train_dl)
        avg_recon_loss = recon_epoch_loss / len(train_dl)
        avg_vq_loss = vq_epoch_loss / len(train_dl)
        
        print(f"Epoch {epoch+1}: total loss={avg_loss:.6f}, reconstruct loss={avg_recon_loss:.6f}, VQ loss={avg_vq_loss:.6f}")
        
        # 每隔save_every轮或最后一轮保存模型
        if (epoch + 1) % opt.save_every == 0 or epoch == opt.epochs - 1:
            save_path = os.path.join(opt.output_dir, f'vqvae_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"model saved to {save_path}")
        
        # 验证
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
            
            print(f"eval: total loss={val_loss:.6f}, reconstruct loss={val_recon_loss:.6f}, VQ loss={val_vq_loss:.6f}")
            
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                best_path = os.path.join(opt.output_dir, 'vqvae_best.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, best_path)
                print(f"best model saved to {best_path}")

    print("training complete!")
    return model

def main():
    opt = parse_args()
    
    if opt.debug:
        torch.autograd.set_detect_anomaly(True)
        print("调试模式已启用")
    
    train_dl, test_dl, test_dl_for_eval = CreateDataLoader(opt)
    print(f"Train set size: {len(train_dl)}, Test set size: {len(test_dl)}")
    
    model = train_vqvae(train_dl, test_dl, opt)
    
    return model

if __name__ == "__main__":
    main()