"""
LIENet Test / Inference Script
Đọc ảnh thiếu sáng, tăng cường bằng model đã train, và lưu kết quả.
"""

import os
import argparse
import time

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config import Config
from model import LIENet


def parse_args():
    parser = argparse.ArgumentParser(description='LIENet Inference')
    parser.add_argument('--test_dir', type=str, default=Config.test_dir,
                        help='Thư mục chứa ảnh test')
    parser.add_argument('--result_dir', type=str, default=Config.result_dir,
                        help='Thư mục lưu kết quả')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/lienet_best.pth',
                        help='Đường dẫn checkpoint model')
    parser.add_argument('--img_size', type=int, default=Config.img_size,
                        help='Kích thước resize ảnh đầu vào')
    parser.add_argument('--original_size', action='store_true',
                        help='Giữ nguyên kích thước ảnh gốc thay vì resize')
    return parser.parse_args()


SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def load_model(checkpoint_path, device):
    """Load model từ checkpoint."""
    model = LIENet(gamma=Config.gamma).to(device)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch = checkpoint.get('epoch', -1)
    loss = checkpoint.get('loss', -1)
    print(f"[INFO] Đã tải model từ epoch {epoch + 1} (loss={loss:.4f})")

    return model


def enhance_single_image(model, img_path, img_size, device, keep_original_size=False):
    """
    Tăng cường một ảnh duy nhất, sử dụng Letterbox Resize.

    Args:
        model: Model LIENet đã load
        img_path: Đường dẫn ảnh
        img_size: Kích thước resize (letterbox)
        device: Thiết bị (CPU/GPU)
        keep_original_size: Giữ nguyên kích thước gốc (loại bỏ padding letterbox)

    Returns:
        result_image: PIL Image đã tăng cường
    """
    from dataset import letterbox_resize

    # Đọc ảnh
    original = Image.open(img_path).convert('RGB')
    orig_w, orig_h = original.size  # (W, H)

    # Letterbox Resize: giữ nguyên tỉ lệ, pad phần thừa
    letterboxed = letterbox_resize(original, img_size, fill=0)

    # Chuyển sang tensor
    to_tensor = transforms.ToTensor()
    input_tensor = to_tensor(letterboxed).unsqueeze(0).to(device)  # (1, 3, H, W)

    # Inference
    with torch.no_grad():
        enhanced, A = model(input_tensor)

    # Chuyển về PIL Image
    enhanced = enhanced.squeeze(0).cpu()  # (3, H, W)
    enhanced = torch.clamp(enhanced, 0.0, 1.0)

    # Chuyển tensor → PIL
    to_pil = transforms.ToPILImage()
    result_image = to_pil(enhanced)

    # Khôi phục kích thước gốc: cắt bỏ padding letterbox rồi resize
    if keep_original_size:
        if isinstance(img_size, int):
            target_h, target_w = img_size, img_size
        else:
            target_h, target_w = img_size

        # Tính lại vùng ảnh thực (không padding) trong letterbox
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2

        # Crop vùng ảnh thực, bỏ padding
        result_image = result_image.crop((pad_left, pad_top,
                                          pad_left + new_w, pad_top + new_h))
        # Resize về kích thước gốc
        result_image = result_image.resize((orig_w, orig_h), Image.BILINEAR)

    return result_image


def test():
    args = parse_args()

    # ========================
    # Thiết lập
    # ========================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Thiết bị: {device}")

    os.makedirs(args.result_dir, exist_ok=True)

    # ========================
    # Load model
    # ========================
    model = load_model(args.checkpoint, device)

    # ========================
    # Lấy danh sách ảnh test
    # ========================
    if not os.path.isdir(args.test_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục test: {args.test_dir}")

    image_files = sorted([
        f for f in os.listdir(args.test_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])

    if len(image_files) == 0:
        raise RuntimeError(f"Không tìm thấy ảnh trong: {args.test_dir}")

    print(f"[INFO] Số ảnh test: {len(image_files)}")
    print(f"[INFO] Kết quả sẽ lưu tại: {args.result_dir}")

    # ========================
    # Inference
    # ========================
    total_time = 0.0

    for filename in tqdm(image_files, desc="Enhancing"):
        img_path = os.path.join(args.test_dir, filename)

        start = time.time()
        result = enhance_single_image(
            model, img_path, args.img_size, device,
            keep_original_size=args.original_size
        )
        elapsed = time.time() - start
        total_time += elapsed

        # Lưu kết quả
        name, ext = os.path.splitext(filename)
        save_path = os.path.join(args.result_dir, f"{name}_enhanced{ext}")
        result.save(save_path)

    avg_time = total_time / len(image_files) if image_files else 0
    print(f"\n[INFO] Hoàn tất!")
    print(f"[INFO] Tổng thời gian: {total_time:.2f}s")
    print(f"[INFO] Trung bình: {avg_time:.3f}s/ảnh")
    print(f"[INFO] Kết quả: {args.result_dir}")


if __name__ == '__main__':
    test()
