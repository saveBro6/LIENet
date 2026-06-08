"""
LIENet Configuration
Tập trung tất cả siêu tham số huấn luyện và kiến trúc mô hình.
"""

import os


class Config:
    # ========================
    # Đường dẫn dữ liệu
    # ========================
    train_dir = "dataset/train"
    test_dir = "dataset/test"
    checkpoint_dir = "checkpoints"
    result_dir = "results"

    # ========================
    # Kiến trúc mô hình
    # ========================
    gamma = 4  # Hệ số gamma cố định cho Dual-Gamma Curve

    # ========================
    # Siêu tham số huấn luyện
    # ========================
    img_size = 640          # Kích thước đầu vào (resize về img_size x img_size)
    epochs = 90             # Số chu kỳ huấn luyện
    batch_size = 8          # Kích thước lô
    lr = 1e-4               # Tốc độ học (Learning Rate)
    num_workers = 4         # Số worker cho DataLoader

    # ========================
    # Trọng số hàm mất mát
    # ========================
    w_spa = 1.0             # Trọng số Spatial Consistency Loss
    w_exp = 1.0             # Trọng số Exposure Control Loss
    w_col = 0.5             # Trọng số Color Constancy Loss
    w_tvA = 20.0            # Trọng số Illuminance Smoothness Loss

    # ========================
    # Tham số hàm mất mát
    # ========================
    exposure_target = 0.6   # Mức phơi sáng lý tưởng (E)
    spa_pool_size = 4       # Kích thước vùng cục bộ cho L_spa
    exp_patch_size = 16     # Kích thước patch cho L_exp (16x16)

    # ========================
    # Logging & Checkpointing
    # ========================
    print_every = 10        # In loss mỗi N batch
    save_every = 5          # Lưu checkpoint mỗi N epoch

    @classmethod
    def ensure_dirs(cls):
        """Tạo các thư mục cần thiết nếu chưa tồn tại."""
        os.makedirs(cls.checkpoint_dir, exist_ok=True)
        os.makedirs(cls.result_dir, exist_ok=True)
