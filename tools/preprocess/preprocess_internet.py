import os
import glob

# split_video_frame_sample10k 12844imgs

# 定义测试数据集的根目录（相对路径）
relative_test_dir = "./data/IndoorGau_dataset/split_video_frame_sample10k"
# 定义输出 txt 文件的路径
output_file = os.path.join(relative_test_dir, "image_paths.txt")

# 定义要查找的图片格式
image_extensions = ["*.jpg", "*.png"]

# 存储所有图片路径
image_paths = []

# 遍历所有图片格式
for ext in image_extensions:
    # 使用 glob 查找所有匹配的图片文件，包括子目录
    pattern = os.path.join(relative_test_dir, "**", ext)
    found_images = glob.glob(pattern, recursive=True)
    image_paths.extend(found_images)

# 将路径排序，确保顺序一致
image_paths.sort()

# 将绝对路径转换为相对路径
relative_paths = []
for path in image_paths:
    # 确保路径以 ./ 开头
    if not path.startswith("./"):
        path = "./" + path
    relative_paths.append(path)

# 将图片路径写入 txt 文件
with open(output_file, "w") as f:
    for path in relative_paths:
        f.write(path + "\n")

print(f"Successfully extracted {len(relative_paths)} image paths.")
print(f"Saved to {output_file}.")
