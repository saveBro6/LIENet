"""
LIENet Training Script
Huấn luyện mô hình LIENet với 4 hàm mất mát zero-reference.
"""

import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from model import LIENet
from losses import TotalLoss
from dataset import LowLightDataset


def parse_args():
    """Parse command line arguments, sử dụng Config làm giá trị mặc định."""
    parser = argparse.ArgumentParser(description='LIENet Training')
    parser.add_argument('--train_dir', type=str, default=Config.train_dir,
                        help='Thư mục chứa ảnh huấn luyện')
    parser.add_argument('--epochs', type=int, default=Config.epochs,
                        help='Số epoch huấn luyện')
    parser.add_argument('--batch_size', type=int, default=Config.batch_size,
                        help='Kích thước batch')
    parser.add_argument('--lr', type=float, default=Config.lr,
                        help='Tốc độ học')
    parser.add_argument('--img_size', type=int, default=Config.img_size,
                        help='Kích thước ảnh đầu vào')
    parser.add_argument('--checkpoint_dir', type=str, default=Config.checkpoint_dir,
                        help='Thư mục lưu checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Đường dẫn checkpoint để tiếp tục huấn luyện')
    parser.add_argument('--num_workers', type=int, default=Config.num_workers,
                        help='Số worker cho DataLoader')
    parser.add_argument('--q', action='store_true',
                        help='Quiet mode: tắt thanh tiến trình tqdm, chỉ hiển thị kết quả cuối epoch')
    return parser.parse_args()


def train():
    args = parse_args()

    # ========================
    # Thiết lập
    # ========================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Thiết bị: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ========================
    # Dataset & DataLoader
    # ========================
    print(f"[INFO] Đang tải dữ liệu từ: {args.train_dir}")
    train_dataset = LowLightDataset(
        img_dir=args.train_dir,
        img_size=args.img_size,
        augment=True  # Bật data augmentation cho training
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print(f"[INFO] Số lượng ảnh huấn luyện: {len(train_dataset)}")
    print(f"[INFO] Số batch/epoch: {len(train_loader)}")

    # ========================
    # Model
    # ========================
    model = LIENet(gamma=Config.gamma).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Tổng tham số: {total_params:,}")
    print(f"[INFO] Tham số huấn luyện: {trainable_params:,}")

    # ========================
    # Optimizer
    # ========================
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ========================
    # Loss Function
    # ========================
    criterion = TotalLoss(
        w_spa=Config.w_spa,
        w_exp=Config.w_exp,
        w_col=Config.w_col,
        w_tvA=Config.w_tvA,
        spa_pool_size=Config.spa_pool_size,
        exp_patch_size=Config.exp_patch_size,
        exposure_target=Config.exposure_target
    ).to(device)

    # ========================
    # Resume từ checkpoint (nếu có)
    # ========================
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(f"[INFO] Đang tải checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"[INFO] Tiếp tục từ epoch {start_epoch}")
        else:
            print(f"[WARN] Không tìm thấy checkpoint: {args.resume}")

    # ========================
    # Training Loop
    # ========================
    print(f"\n{'='*60}")
    print(f"  BẮT ĐẦU HUẤN LUYỆN LIENet")
    print(f"  Epochs: {start_epoch} → {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Gamma: {Config.gamma}")
    print(f"  Loss weights: spa={Config.w_spa}, exp={Config.w_exp}, "
          f"col={Config.w_col}, tvA={Config.w_tvA}")
    print(f"{'='*60}\n")

    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {
            'L_spa': 0.0, 'L_exp': 0.0,
            'L_col': 0.0, 'L_tvA': 0.0,
            'L_total': 0.0
        }

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch [{epoch+1}/{args.epochs}]",
            ncols=120,
            disable=args.q
        )

        start_time = time.time()

        for batch_idx, (images, _) in pbar:
            images = images.to(device)

            # Forward pass
            enhanced, A = model(images)

            # Tính loss
            total_loss, loss_dict = criterion(enhanced, images, A)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Cập nhật thống kê
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]

            # Cập nhật thanh tiến trình
            if not args.q and ((batch_idx + 1) % Config.print_every == 0 or batch_idx == 0):
                pbar.set_postfix({
                    'total': f"{loss_dict['L_total']:.4f}",
                    'spa': f"{loss_dict['L_spa']:.4f}",
                    'exp': f"{loss_dict['L_exp']:.4f}",
                    'col': f"{loss_dict['L_col']:.4f}",
                    'tvA': f"{loss_dict['L_tvA']:.4f}",
                })

        elapsed = time.time() - start_time
        num_batches = len(train_loader)

        # Tính trung bình loss cho epoch
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        # In tổng kết epoch
        print(f"\n  Epoch [{epoch+1}/{args.epochs}] - {elapsed:.1f}s")
        print(f"  Loss: total={epoch_losses['L_total']:.4f} | "
              f"spa={epoch_losses['L_spa']:.4f} | "
              f"exp={epoch_losses['L_exp']:.4f} | "
              f"col={epoch_losses['L_col']:.4f} | "
              f"tvA={epoch_losses['L_tvA']:.4f}")

        # Ghi log kết quả epoch vào train.log ở checkpoints/
        log_path = os.path.join(args.checkpoint_dir, 'train.log')
        log_mode = 'w' if (epoch == 0 and start_epoch == 0) else 'a'
        with open(log_path, log_mode, encoding='utf-8') as f:
            f.write(f"Epoch [{epoch+1}/{args.epochs}] - {elapsed:.1f}s - "
                    f"Loss: total={epoch_losses['L_total']:.4f} | "
                    f"spa={epoch_losses['L_spa']:.4f} | "
                    f"exp={epoch_losses['L_exp']:.4f} | "
                    f"col={epoch_losses['L_col']:.4f} | "
                    f"tvA={epoch_losses['L_tvA']:.4f}\n")

        # ========================
        # Lưu checkpoint
        # ========================
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_losses['L_total'],
        }

        # Lưu checkpoint định kỳ
        if (epoch + 1) % Config.save_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f'lienet_epoch_{epoch+1}.pth')
            torch.save(checkpoint_data, ckpt_path)
            print(f"  [SAVE] Checkpoint: {ckpt_path}")

        # Lưu best model
        if epoch_losses['L_total'] < best_loss:
            best_loss = epoch_losses['L_total']
            best_path = os.path.join(args.checkpoint_dir, 'lienet_best.pth')
            torch.save(checkpoint_data, best_path)
            print(f"  [BEST] Best model saved (loss={best_loss:.4f})")

        # Luôn lưu checkpoint cuối cùng
        last_path = os.path.join(args.checkpoint_dir, 'lienet_last.pth')
        torch.save(checkpoint_data, last_path)

        print()

    print(f"{'='*60}")
    print(f"  HOÀN TẤT HUẤN LUYỆN!")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    train()
