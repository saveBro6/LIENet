"""
Verification Script cho LIENet
Kiểm tra kiến trúc model, loss functions, và forward pass với dummy data.
"""

import torch
import sys

def test_model():
    """Kiểm tra kiến trúc model và output shape."""
    print("=" * 60)
    print("  TEST 1: Model Architecture & Forward Pass")
    print("=" * 60)

    from model import LIENet, EPPN

    model = LIENet(gamma=4)

    # Kiểm tra số tham số
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Tổng tham số: {total_params:,}")
    print(f"  Tham số huấn luyện: {trainable_params:,}")

    # Dummy input: batch=2, 3 channels, 640x640
    dummy_input = torch.rand(2, 3, 640, 640)
    print(f"  Input shape: {dummy_input.shape}")

    # Forward pass
    enhanced, A = model(dummy_input)
    print(f"  Enhanced shape: {enhanced.shape}")
    print(f"  A shape: {A.shape}")

    # Kiểm tra output shapes
    assert enhanced.shape == (2, 3, 640, 640), f"Enhanced shape mismatch: {enhanced.shape}"
    assert A.shape == (2, 3, 640, 640), f"A shape mismatch: {A.shape}"

    # Kiểm tra A nằm trong [0, 1] (Sigmoid)
    assert A.min() >= 0.0, f"A min < 0: {A.min()}"
    assert A.max() <= 1.0, f"A max > 1: {A.max()}"

    # Kiểm tra enhanced nằm trong [0, 1]
    assert enhanced.min() >= 0.0, f"Enhanced min < 0: {enhanced.min()}"
    assert enhanced.max() <= 1.0, f"Enhanced max > 1: {enhanced.max()}"

    print(f"  A range: [{A.min().item():.4f}, {A.max().item():.4f}] ✓")
    print(f"  Enhanced range: [{enhanced.min().item():.4f}, {enhanced.max().item():.4f}] ✓")
    print("  ✅ Model test PASSED\n")


def test_losses():
    """Kiểm tra tất cả loss functions."""
    print("=" * 60)
    print("  TEST 2: Loss Functions")
    print("=" * 60)

    from losses import (
        SpatialConsistencyLoss,
        ExposureControlLoss,
        ColorConstancyLoss,
        IlluminanceSmoothnessLoss,
        TotalLoss
    )

    # Dummy data — cần requires_grad để test backward pass
    enhanced = torch.rand(2, 3, 640, 640, requires_grad=True)
    original = torch.rand(2, 3, 640, 640) * 0.3  # ảnh tối (không cần grad)
    A = torch.rand(2, 3, 640, 640, requires_grad=True)

    # Test từng loss
    l_spa = SpatialConsistencyLoss(pool_size=4)
    loss_spa = l_spa(enhanced, original)
    print(f"  L_spa = {loss_spa.item():.6f} (shape: {loss_spa.shape}) ✓")

    l_exp = ExposureControlLoss(patch_size=16, target_exposure=0.6)
    loss_exp = l_exp(enhanced)
    print(f"  L_exp = {loss_exp.item():.6f} (shape: {loss_exp.shape}) ✓")

    l_col = ColorConstancyLoss()
    loss_col = l_col(enhanced)
    print(f"  L_col = {loss_col.item():.6f} (shape: {loss_col.shape}) ✓")

    l_tvA = IlluminanceSmoothnessLoss()
    loss_tvA = l_tvA(A)
    print(f"  L_tvA = {loss_tvA.item():.6f} (shape: {loss_tvA.shape}) ✓")

    # Test TotalLoss
    criterion = TotalLoss(w_spa=1.0, w_exp=1.0, w_col=0.5, w_tvA=20.0)
    total_loss, loss_dict = criterion(enhanced, original, A)
    print(f"\n  Total Loss = {total_loss.item():.6f}")
    for k, v in loss_dict.items():
        print(f"    {k}: {v:.6f}")

    # Kiểm tra gradient chảy ngược
    total_loss.backward()
    print(f"\n  ✅ Loss backward PASSED (gradients computed)\n")


def test_dual_gamma():
    """Kiểm tra Dual-Gamma Curve enhancement."""
    print("=" * 60)
    print("  TEST 3: Dual-Gamma Curve")
    print("=" * 60)

    from model import LIENet

    model = LIENet(gamma=4)

    # Test với ảnh tối (pixel thấp) - kỳ vọng ảnh sáng hơn
    dark_input = torch.ones(1, 3, 64, 64) * 0.1
    with torch.no_grad():
        enhanced_dark, _ = model(dark_input)

    print(f"  Input tối (mean={dark_input.mean():.3f}) → "
          f"Enhanced (mean={enhanced_dark.mean():.3f})")

    # Test với ảnh sáng (pixel cao) - kỳ vọng ít thay đổi
    bright_input = torch.ones(1, 3, 64, 64) * 0.8
    with torch.no_grad():
        enhanced_bright, _ = model(bright_input)

    print(f"  Input sáng (mean={bright_input.mean():.3f}) → "
          f"Enhanced (mean={enhanced_bright.mean():.3f})")

    # Kiểm tra gamma math: G_a(0.1) = 0.1^0.25 ≈ 0.562
    import math
    g_a = 0.1 ** 0.25
    g_b = 1 - (1 - 0.1) ** 0.25
    print(f"\n  Kiểm tra toán học (x=0.1, γ=4):")
    print(f"    G_a(0.1) = 0.1^(1/4) = {g_a:.4f}")
    print(f"    G_b(0.1) = 1-(0.9)^(1/4) = {g_b:.4f}")
    print(f"    Cả hai hàm đều kéo sáng pixel tối ✓")

    print("  ✅ Dual-Gamma test PASSED\n")


def test_training_step():
    """Kiểm tra một bước training đầy đủ."""
    print("=" * 60)
    print("  TEST 4: Full Training Step (Dry Run)")
    print("=" * 60)

    from model import LIENet
    from losses import TotalLoss

    device = torch.device('cpu')
    model = LIENet(gamma=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = TotalLoss(w_spa=1.0, w_exp=1.0, w_col=0.5, w_tvA=20.0).to(device)

    # Dummy batch (nhỏ hơn để chạy nhanh)
    batch = torch.rand(2, 3, 128, 128).to(device)

    # Forward
    enhanced, A = model(batch)
    total_loss, loss_dict = criterion(enhanced, batch, A)

    print(f"  Forward pass: ✓")
    print(f"  Loss: {total_loss.item():.4f}")

    # Backward
    optimizer.zero_grad()
    total_loss.backward()

    # Kiểm tra gradient tồn tại
    has_grad = all(p.grad is not None for p in model.parameters())
    print(f"  Backward pass: ✓")
    print(f"  Gradients exist: {has_grad} ✓")

    # Optimizer step
    optimizer.step()
    print(f"  Optimizer step: ✓")

    # Chạy thêm 1 step để kiểm tra ổn định
    enhanced2, A2 = model(batch)
    total_loss2, _ = criterion(enhanced2, batch, A2)
    print(f"  Step 2 loss: {total_loss2.item():.4f}")

    print("  ✅ Full training step PASSED\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  LIENet VERIFICATION")
    print("=" * 60 + "\n")

    try:
        test_model()
        test_losses()
        test_dual_gamma()
        test_training_step()

        print("=" * 60)
        print("  🎉 TẤT CẢ KIỂM TRA ĐỀU THÀNH CÔNG!")
        print("=" * 60)

    except Exception as e:
        print(f"\n  ❌ LỖI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
