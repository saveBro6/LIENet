"""
LIENet Loss Functions
4 hàm mất mát zero-reference (không cần ảnh tham chiếu) cho LIENet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialConsistencyLoss(nn.Module):
    """
    Mất mát tính nhất quán không gian (L_spa).

    Giữ độ chênh lệch cường độ của các vùng lân cận trên ảnh đầu ra
    giống với ảnh gốc.

    Công thức:
        L_spa = (1/S) * Σ_i Σ_{j∈Ω(i)} (|Y_i - Y_j| - |I_i - I_j|)^2

    Trong đó:
        S: Số lượng vùng cục bộ
        Ω(i): 4 vùng lân cận kề (trên, dưới, trái, phải)
        Y, I: Cường độ trung bình vùng của ảnh tăng cường và ảnh gốc
    """

    def __init__(self, pool_size=4):
        super(SpatialConsistencyLoss, self).__init__()
        # Kernel trích xuất chênh lệch với 4 vùng lân cận
        # Trái
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        # Phải
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        # Trên
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        # Dưới
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)

        self.register_buffer('kernel_left', kernel_left)
        self.register_buffer('kernel_right', kernel_right)
        self.register_buffer('kernel_up', kernel_up)
        self.register_buffer('kernel_down', kernel_down)

        self.pool = nn.AvgPool2d(pool_size)

    def forward(self, enhanced, original):
        """
        Args:
            enhanced: Ảnh đã tăng cường (B, 3, H, W)
            original: Ảnh gốc thiếu sáng (B, 3, H, W)

        Returns:
            loss: Scalar tensor
        """
        # Chuyển sang cường độ trung bình (grayscale) rồi chia vùng
        enhanced_mean = torch.mean(enhanced, dim=1, keepdim=True)  # (B, 1, H, W)
        original_mean = torch.mean(original, dim=1, keepdim=True)  # (B, 1, H, W)

        # Lấy cường độ trung bình của các vùng cục bộ
        enhanced_pool = self.pool(enhanced_mean)  # (B, 1, H/k, W/k)
        original_pool = self.pool(original_mean)  # (B, 1, H/k, W/k)

        # Tính chênh lệch với 4 vùng lân cận
        d_enhanced_left = F.conv2d(enhanced_pool, self.kernel_left, padding=1)
        d_enhanced_right = F.conv2d(enhanced_pool, self.kernel_right, padding=1)
        d_enhanced_up = F.conv2d(enhanced_pool, self.kernel_up, padding=1)
        d_enhanced_down = F.conv2d(enhanced_pool, self.kernel_down, padding=1)

        d_original_left = F.conv2d(original_pool, self.kernel_left, padding=1)
        d_original_right = F.conv2d(original_pool, self.kernel_right, padding=1)
        d_original_up = F.conv2d(original_pool, self.kernel_up, padding=1)
        d_original_down = F.conv2d(original_pool, self.kernel_down, padding=1)

        # L_spa = mean( (|Y_i - Y_j| - |I_i - I_j|)^2 )
        loss = (
            torch.pow(d_enhanced_left.abs() - d_original_left.abs(), 2)
            + torch.pow(d_enhanced_right.abs() - d_original_right.abs(), 2)
            + torch.pow(d_enhanced_up.abs() - d_original_up.abs(), 2)
            + torch.pow(d_enhanced_down.abs() - d_original_down.abs(), 2)
        )

        return torch.mean(loss)


class ExposureControlLoss(nn.Module):
    """
    Mất mát kiểm soát phơi sáng (L_exp).

    Kiểm soát vùng sáng không bị cháy, đưa cường độ trung bình mỗi vùng
    về gần mức phơi sáng lý tưởng E.

    Công thức:
        L_exp = (1/M) * Σ_k |Y_k - E|

    Trong đó:
        M: Số vùng không chồng chéo 16x16
        Y_k: Cường độ trung bình vùng k
        E = 0.6: Mức phơi sáng lý tưởng
    """

    def __init__(self, patch_size=16, target_exposure=0.6):
        super(ExposureControlLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.target_exposure = target_exposure

    def forward(self, enhanced):
        """
        Args:
            enhanced: Ảnh đã tăng cường (B, 3, H, W)

        Returns:
            loss: Scalar tensor
        """
        # Chuyển sang cường độ trung bình
        enhanced_mean = torch.mean(enhanced, dim=1, keepdim=True)  # (B, 1, H, W)

        # Chia thành các vùng 16x16 không chồng chéo
        patches = self.pool(enhanced_mean)  # (B, 1, H/16, W/16)

        # L_exp = mean(|Y_k - E|)
        loss = torch.mean(torch.abs(patches - self.target_exposure))

        return loss


class ColorConstancyLoss(nn.Module):
    """
    Mất mát độ ổn định màu (L_col).

    Điều chỉnh cân bằng màu sắc trên 3 kênh RGB để tránh
    hiện tượng ám màu (color cast).

    Công thức:
        L_col = Σ_{(p,q)∈ε} (J_p - J_q)^2
        ε = {(R,G), (R,B), (G,B)}

    Trong đó:
        J_p, J_q: Cường độ trung bình của kênh màu p, q
    """

    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, enhanced):
        """
        Args:
            enhanced: Ảnh đã tăng cường (B, 3, H, W)

        Returns:
            loss: Scalar tensor
        """
        # Tính cường độ trung bình mỗi kênh trên toàn ảnh
        mean_rgb = torch.mean(enhanced, dim=[2, 3], keepdim=True)  # (B, 3, 1, 1)
        mean_r = mean_rgb[:, 0, :, :]  # (B, 1, 1)
        mean_g = mean_rgb[:, 1, :, :]  # (B, 1, 1)
        mean_b = mean_rgb[:, 2, :, :]  # (B, 1, 1)

        # L_col = (J_R - J_G)^2 + (J_R - J_B)^2 + (J_G - J_B)^2
        loss = (
            torch.pow(mean_r - mean_g, 2)
            + torch.pow(mean_r - mean_b, 2)
            + torch.pow(mean_g - mean_b, 2)
        )

        return torch.mean(loss)


class IlluminanceSmoothnessLoss(nn.Module):
    """
    Mất mát độ mịn ánh sáng (L_tvA).

    Giữ tỷ lệ chuyển đổi độ sáng mượt mà trên các kênh,
    ngăn chặn hiện tượng chuyển đổi đột ngột.

    Công thức:
        L_tvA = (1/N) * Σ_n Σ_{c∈δ} (|∇x A_n^c| + |∇y A_n^c|)^2
        δ = {R, G, B}

    Trong đó:
        ∇x, ∇y: Đạo hàm bậc nhất theo chiều ngang/dọc
        N: Số lần lặp (= 1 cho LIENet single-pass)
    """

    def __init__(self):
        super(IlluminanceSmoothnessLoss, self).__init__()

    def forward(self, A):
        """
        Args:
            A: Ma trận tham số tăng cường (B, 3, H, W)

        Returns:
            loss: Scalar tensor
        """
        # Đạo hàm theo chiều ngang (∇x): so sánh pixel liền kề theo chiều W
        diff_x = A[:, :, :, :-1] - A[:, :, :, 1:]  # (B, 3, H, W-1)
        # Đạo hàm theo chiều dọc (∇y): so sánh pixel liền kề theo chiều H
        diff_y = A[:, :, :-1, :] - A[:, :, 1:, :]  # (B, 3, H-1, W)

        # (|∇x| + |∇y|)^2 — cần cùng kích thước nên cắt về (H-1, W-1)
        loss = torch.mean(
            torch.pow(diff_x[:, :, :-1, :].abs() + diff_y[:, :, :, :-1].abs(), 2)
        )

        return loss


class TotalLoss(nn.Module):
    """
    Hàm mất mát tổng quát cho LIENet.

    L_total = W_spa * L_spa + W_exp * L_exp + W_col * L_col + W_tvA * L_tvA

    Mặc định: W_spa = 1, W_exp = 1, W_col = 0.5, W_tvA = 20
    """

    def __init__(self, w_spa=1.0, w_exp=1.0, w_col=0.5, w_tvA=20.0,
                 spa_pool_size=4, exp_patch_size=16, exposure_target=0.6):
        super(TotalLoss, self).__init__()

        self.w_spa = w_spa
        self.w_exp = w_exp
        self.w_col = w_col
        self.w_tvA = w_tvA

        self.L_spa = SpatialConsistencyLoss(pool_size=spa_pool_size)
        self.L_exp = ExposureControlLoss(patch_size=exp_patch_size,
                                         target_exposure=exposure_target)
        self.L_col = ColorConstancyLoss()
        self.L_tvA = IlluminanceSmoothnessLoss()

    def forward(self, enhanced, original, A):
        """
        Args:
            enhanced: Ảnh đã tăng cường (B, 3, H, W)
            original: Ảnh gốc thiếu sáng (B, 3, H, W)
            A: Ma trận tham số dự đoán (B, 3, H, W)

        Returns:
            total_loss: Scalar tensor (gọi .backward() trên giá trị này)
            loss_dict: Dictionary chứa từng thành phần loss
        """
        loss_spa = self.L_spa(enhanced, original)
        loss_exp = self.L_exp(enhanced)
        loss_col = self.L_col(enhanced)
        loss_tvA = self.L_tvA(A)

        total_loss = (
            self.w_spa * loss_spa
            + self.w_exp * loss_exp
            + self.w_col * loss_col
            + self.w_tvA * loss_tvA
        )

        loss_dict = {
            'L_spa': loss_spa.item(),
            'L_exp': loss_exp.item(),
            'L_col': loss_col.item(),
            'L_tvA': loss_tvA.item(),
            'L_total': total_loss.item(),
        }

        return total_loss, loss_dict
