import os
import random
import shutil

# --- 1. Định nghĩa đường dẫn nguồn ---
current_data_dir = "/media/hungmtp/DATA1/YOLO-mines/dataset_640_from_1536_copy2/images"  # Thư mục chứa sẵn train/ và test/
train_src_dir = os.path.join(current_data_dir, "train")
test_src_dir = os.path.join(current_data_dir, "test")

# --- 2. Định nghĩa thư mục đầu ra (output_folder) ---
output_folder = "dataset/"  # Thư mục đích mới của bạn
train_dst_dir = os.path.join(output_folder, "train")
test_dst_dir = os.path.join(output_folder, "test")

# Tự động tạo cấu trúc thư mục đích mới nếu chưa có
os.makedirs(train_dst_dir, exist_ok=True)
os.makedirs(test_dst_dir, exist_ok=True)


# --- 3. Hàm lấy danh sách file và bốc mẫu ngẫu nhiên ---
def sample_and_copy(src_dir, dst_dir, sample_size):
    # Lấy toàn bộ file hợp lệ (bỏ qua file ẩn hệ thống như .DS_Store)
    all_files = [
        f
        for f in os.listdir(src_dir)
        if os.path.isfile(os.path.join(src_dir, f)) and not f.startswith(".")
    ]

    total_files = len(all_files)
    print(f"Thư mục nguồn '{os.path.basename(src_dir)}' có tổng cộng: {total_files} file.")

    # Giới hạn số lượng lấy nếu tổng số file ít hơn yêu cầu
    actual_sample_size = min(sample_size, total_files)
    if actual_sample_size < sample_size:
        print(
            f"⚠️ Cảnh báo: Số lượng file không đủ! Chỉ lấy được tối đa {actual_sample_size}/{sample_size} file."
        )

    # Bốc ngẫu nhiên
    sampled_files = random.sample(all_files, actual_sample_size)

    # Thực hiện sao chép file
    print(f"-> Đang sao chép {actual_sample_size} file sang {dst_dir}...")
    for file_name in sampled_files:
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        shutil.copy2(src_path, dst_path)

    print(f"✅ Hoàn thành phân đoạn: {os.path.basename(src_dir)}\n")


# --- 4. Chạy tiến trình ---
print("--- BẮT ĐẦU QUÁ TRÌNH TRÍCH XUẤT DỮ LIỆU ---\n")

# Bốc 1200 ảnh từ tập train
sample_and_copy(train_src_dir, train_dst_dir, sample_size=1200)

# Bốc 120 ảnh từ tập test
sample_and_copy(test_src_dir, test_dst_dir, sample_size=120)

print("--- TẤT CẢ ĐÃ HOÀN THÀNH ---")


# import os
# import random
# import re
# import shutil

# # --- 1. Định nghĩa đường dẫn ---
# input_folder = "/media/hungmtp/DATA1/YOLO-mines/person_1"  # Thay bằng đường dẫn thư mục gốc của bạn
# output_train_folder = "dataset/train"  # Thư mục lưu tập train
# output_test_folder = "dataset/test"  # Thư mục lưu tập test

# # Tạo thư mục đầu ra nếu chưa có
# os.makedirs(output_train_folder, exist_ok=True)
# os.makedirs(output_test_folder, exist_ok=True)


# # --- 2. Hàm lấy số đầu tiên của thư mục để sort ---
# def get_leading_number(folder_name):
#     match = re.match(r"^(\d+)", folder_name)
#     return int(match.group(1)) if match else float("inf")


# # --- 3. Lọc và sắp xếp các thư mục con ---
# # Chỉ lấy các thư mục có số bắt đầu từ 128 đến 208, hoặc thư mục test 305
# subfolders = []
# for f in os.listdir(input_folder):
#     full_path = os.path.join(input_folder, f)
#     if os.path.isdir(full_path):
#         num = get_leading_number(f)
#         if (128 <= num <= 208) or (num == 305):
#             subfolders.append(f)

# # Sắp xếp từ thấp đến cao theo số đầu tiên
# subfolders.sort(key=get_leading_number)

# # --- 4. Xử lý bốc random 100 frames và export ---
# for folder_name in subfolders:
#     src_folder_path = os.path.join(input_folder, folder_name)

#     # Lấy danh sách tất cả file ảnh trong thư mục con (hỗ trợ .jpg, .jpeg, .png...)
#     all_frames = [
#         f
#         for f in os.listdir(src_folder_path)
#         if f.lower().endswith((".jpg", ".jpeg", ".png"))
#     ]

#     # Kiểm tra số lượng ảnh
#     num_to_sample = min(100, len(all_frames))
#     sampled_frames = random.sample(all_frames, num_to_sample)

#     # Xác định đây là tập Train hay Test
#     if folder_name.startswith("305_"):
#         target_output_dir = output_test_folder
#     else:
#         target_output_dir = output_train_folder

#     print(
#         f"Processing: {folder_name} -> Copying {num_to_sample} frames to {os.path.basename(target_output_dir)}"
#     )

#     # Copy và đổi tên file thành format: ..._sequence_frames_....jpg
#     for frame_name in sampled_frames:
#         # Tách phần mở rộng cũ để lấy tên gốc
#         base_name, _ = os.path.splitext(frame_name)

#         # Tạo tên file mới theo cấu trúc yêu cầu để tránh trùng lặp
#         new_frame_name = f"{folder_name}_frames_{base_name}.jpg"

#         # Đường dẫn file gốc và file đích
#         src_file_path = os.path.join(src_folder_path, frame_name)
#         dst_file_path = os.path.join(target_output_dir, new_frame_name)

#         # Thực hiện copy file
#         shutil.copy2(src_file_path, dst_file_path)

# print("\nHoàn thành việc chia dữ liệu Train/Test!")
