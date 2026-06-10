"""
LIENet Model
Kiến trúc EPPN (Enhancement Parameter Prediction Network) và
thuật toán tăng cường ánh sáng Dual-Gamma Curve.
"""

import torch
import torch.nn as nn


class EPPN(nn.Module):
    """
    Enhancement Parameter Prediction Network.

    Mạng CNN gọn nhẹ dự đoán ma trận tham số tăng cường A
    có cùng kích thước không gian với ảnh đầu vào.

    Kiến trúc:
        - 7 lớp Conv2d (kernel 3x3, padding 1, 32 channels)
        - Lớp 1-4: Sequential (tuần tự) với ReLU
        - Lớp 5-7: Symmetric skip connections (đối xứng ngược)
            + Lớp 5: concat với lớp 3  (in_channels = 64)
            + Lớp 6: concat với lớp 2  (in_channels = 64)
            + Lớp 7: concat với lớp 1  (in_channels = 64)
        - Lớp cuối dùng Sigmoid (output 3 channels cho RGB)
    """

    def __init__(self):
        super(EPPN, self).__init__()

        # === Encoder (4 lớp tuần tự) ===
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        # === Decoder (3 lớp với symmetric skip connections) ===
        # Input channels = 32 (prev) + 32 (skip) = 64
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        # Lớp cuối: output 3 channels (R, G, B) cho ma trận A
        self.conv7 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Ảnh đầu vào (B, 3, H, W), giá trị trong [0, 1]

        Returns:
            A: Ma trận tham số tăng cường (B, 3, H, W), giá trị trong [0, 1]
        """
        # Encoder: 4 lớp tuần tự
        feat1 = self.relu(self.conv1(x))      # (B, 32, H, W)
        feat2 = self.relu(self.conv2(feat1))  # (B, 32, H, W)
        feat3 = self.relu(self.conv3(feat2))  # (B, 32, H, W)
        feat4 = self.relu(self.conv4(feat3))  # (B, 32, H, W)

        # Decoder: 3 lớp với symmetric skip connections (đối xứng ngược)
        # Lớp 5 concat với lớp 3
        feat5 = self.relu(self.conv5(torch.cat([feat4, feat3], dim=1)))  # (B, 32, H, W)
        # Lớp 6 concat với lớp 2
        feat6 = self.relu(self.conv6(torch.cat([feat5, feat2], dim=1)))  # (B, 32, H, W)
        # Lớp 7 concat với lớp 1 → Sigmoid (không dùng ReLU)
        A = self.sigmoid(self.conv7(torch.cat([feat6, feat1], dim=1)))   # (B, 3, H, W)

        return A


class LIENet(nn.Module):
    """
    Low-Light Image Enhancement Network.

    Kết hợp EPPN để dự đoán ma trận tham số A và
    Dual-Gamma Curve để tăng cường độ sáng ảnh.

    Công thức:
        F(X; A) = A · G_a(X) + (1 - A) · G_b(X)
        G_a(X) = X^(1/γ)
        G_b(X) = 1 - (1 - X)^(1/γ)
        γ = 4
    """

    def __init__(self, gamma=4):
        super(LIENet, self).__init__()
        self.eppn = EPPN()
        self.gamma = gamma

    def dual_gamma_curve(self, x, A):
        """
        Áp dụng Dual-Gamma Curve để tăng cường ảnh.

        Args:
            x: Ảnh gốc (B, 3, H, W), giá trị trong [0, 1]
            A: Ma trận tham số (B, 3, H, W), giá trị trong [0, 1]

        Returns:
            enhanced: Ảnh đã tăng cường (B, 3, H, W)
        """
        inv_gamma = 1.0 / self.gamma

        # G_a(X) = X^(1/γ) — kéo giãn vùng tối
        G_a = torch.pow(x + 1e-8, inv_gamma)

        # G_b(X) = 1 - (1 - X)^(1/γ) — nén vùng sáng
        G_b = 1.0 - torch.pow(1.0 - x + 1e-8, inv_gamma)

        # F(X; A) = A · G_a(X) + (1 - A) · G_b(X)
        enhanced = A * G_a + (1.0 - A) * G_b

        # Clamp để đảm bảo giá trị pixel hợp lệ
        enhanced = torch.clamp(enhanced, 0.0, 1.0)

        return enhanced

    def forward(self, x):
        """
        Args:
            x: Ảnh thiếu sáng (B, 3, H, W), giá trị trong [0, 1]

        Returns:
            enhanced: Ảnh đã tăng cường (B, 3, H, W)
            A: Ma trận tham số dự đoán (B, 3, H, W)
        """
        # Bước 1: EPPN dự đoán ma trận tham số A
        A = self.eppn(x)

        # Bước 2: Áp dụng Dual-Gamma Curve
        enhanced = self.dual_gamma_curve(x, A)

        return enhanced, A
