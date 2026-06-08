"""
LIENet Dataset
Dataset class cho ảnh thiếu sáng, không cần ảnh tham chiếu (zero-reference).
Sử dụng Letterbox Resize để giữ nguyên tỉ lệ ảnh gốc.
"""

import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


# Định dạng ảnh được hỗ trợ
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def letterbox_resize(image, target_size, fill=0):
    """
    Resize ảnh theo kiểu Letterbox: giữ nguyên tỉ lệ gốc (aspect ratio),
    co ảnh vừa khít trong target_size, phần thừa được pad đều hai bên.

    Args:
        image: PIL Image
        target_size: int hoặc (h, w) — kích thước đích
        fill: Giá trị pixel để pad (0 = đen, 114 = xám YOLO-style)

    Returns:
        padded: PIL Image với kích thước đúng target_size x target_size
    """
    if isinstance(target_size, int):
        target_h, target_w = target_size, target_size
    else:
        target_h, target_w = target_size

    orig_w, orig_h = image.size  # PIL trả về (W, H)

    # Nếu ảnh đã có đúng kích thước đích, trả về ảnh gốc luôn để tối ưu hiệu năng
    if orig_w == target_w and orig_h == target_h:
        return image

    # Tính tỉ lệ scale giữ nguyên aspect ratio (lấy min để vừa khít)
    scale = min(target_w / orig_w, target_h / orig_h)

    # Kích thước mới sau khi scale
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Resize ảnh giữ nguyên tỉ lệ
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    # Tính padding đều hai bên (top/bottom hoặc left/right)
    pad_w = target_w - new_w  # tổng padding theo chiều ngang
    pad_h = target_h - new_h  # tổng padding theo chiều dọc

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # Tạo ảnh mới với kích thước target, fill bằng màu pad
    if isinstance(fill, (int, float)):
        fill_color = (fill, fill, fill)
    else:
        fill_color = fill

    padded = Image.new('RGB', (target_w, target_h), fill_color)
    padded.paste(resized, (pad_left, pad_top))

    return padded


class LowLightDataset(Dataset):
    """
    Dataset cho ảnh thiếu sáng.

    Chỉ cần cung cấp thư mục chứa ảnh (không cần ground-truth).
    Sử dụng Letterbox Resize để giữ nguyên tỉ lệ ảnh gốc,
    phần thừa được pad đều hai bên.

    Args:
        img_dir: Đường dẫn thư mục chứa ảnh
        img_size: Kích thước resize (mặc định 640)
        augment: Bật/tắt data augmentation (chỉ dùng khi training)
        pad_fill: Giá trị pixel để pad letterbox (0 = đen)
    """

    def __init__(self, img_dir, img_size=640, augment=False, pad_fill=0):
        super(LowLightDataset, self).__init__()
        self.img_dir = img_dir
        self.img_size = img_size
        self.augment = augment
        self.pad_fill = pad_fill

        # Lấy danh sách file ảnh hợp lệ
        self.image_files = sorted([
            f for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ])

        if len(self.image_files) == 0:
            raise RuntimeError(
                f"Không tìm thấy ảnh trong thư mục: {img_dir}\n"
                f"Định dạng hỗ trợ: {SUPPORTED_EXTENSIONS}"
            )

        # Data augmentation cho training
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ])
        else:
            self.augment_transform = None

        # Transform sau letterbox: chỉ cần ToTensor (chuẩn hóa [0,255] → [0,1])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor (3, H, W) với giá trị trong [0, 1], kích thước img_size x img_size
            filename: Tên file ảnh gốc
        """
        filename = self.image_files[idx]
        img_path = os.path.join(self.img_dir, filename)

        # Đọc ảnh và chuyển sang RGB
        image = Image.open(img_path).convert('RGB')

        # Áp dụng augmentation trước resize (nếu có)
        if self.augment_transform is not None:
            image = self.augment_transform(image)

        # Letterbox Resize: giữ nguyên tỉ lệ, pad phần thừa
        image = letterbox_resize(image, self.img_size, fill=self.pad_fill)

        # Chuyển sang tensor [0, 1]
        image = self.to_tensor(image)

        return image, filename
