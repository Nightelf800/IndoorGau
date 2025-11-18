import os
import shutil


# 生成训练集和测试集的索引列表
def get_train_test_indices(sorted_unique):
    """根据排序后的唯一索引列表，生成训练集和测试集的索引列表
    规则：每10帧，前9帧为train，后1帧为test
    """
    train_indices = []
    test_indices = []

    N = len(sorted_unique)

    # 对每10帧进行处理
    for i in range(0, N, split_size):
        # 获取当前20帧的索引范围
        batch_end = min(i + split_size, N)
        batch_indices = sorted_unique[i:batch_end]

        # 最后1帧作为测试集（如果有足够的帧）
        if len(batch_indices) > test_split:
            train_indices.extend(batch_indices[:len(batch_indices) - test_split])
            test_indices.extend(batch_indices[-test_split:])
        elif len(batch_indices) > 0:
            train_indices.extend(batch_indices[:len(batch_indices) - 1])
            test_indices.append(batch_indices[-1])

    return train_indices, test_indices


# 获取排序后的唯一文件索引
def get_sorted_unique_indices(folder_path):
    file_numbers = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 只处理后缀为.pkl的文件
        if filename.endswith(".pkl"):
            # 提取文件名前缀（去掉.pkl），并尝试转为整数
            try:
                # 假设文件名格式是“6位数字.pkl”，如000000.pkl → 提取“000000”
                num_str = filename.split(".")[0]  # 分割文件名，取“.”前面的部分
                num = int(num_str)  # 转为整数（如“000001”→1，不影响排序）
                file_numbers.append(num)
            except:
                # 跳过非“数字.pkl”格式的文件（如abc.pkl）
                print(f"跳过非目标文件：{filename}")

    if not file_numbers:
        print("文件夹中未找到符合格式的.pkl文件！")
        return []

    # 从小到大排序
    sorted_numbers = sorted(file_numbers)
    # 去重（若有重复文件，保留一个）
    sorted_unique = list(set(sorted_numbers))
    sorted_unique.sort()  # 去重后重新排序（确保顺序正确）

    return sorted_unique


def process_honor_files():
    # 源目录和目标目录设置
    train_source_roots = 'data/honor_data_0920/20250711_1519_finterval1'
    test_source_roots = 'data/honor_data_0920/20250711_1540_finterval1'
    source_roots = [
        'data/honor_data_0920/20250711_1519_finterval1',
        'data/honor_data_0920/20250711_1540_finterval1'
    ]
    # 保存文件路径到txt文件
    output_dir = './data/honor_data_0920/'
    os.makedirs(output_dir, exist_ok=True)

    preprocess_folders = ['voxels']
    train_files = []
    test_files = []

    # 处理每个源文件夹
    for idx, source_root in enumerate(source_roots):
        # 获取源文件夹名称作为前缀
        source_name = os.path.basename(source_root)

        # 获取当前源文件夹的排序后唯一索引（使用voxels文件夹中的pkl文件）
        voxels_path = os.path.join(source_root, preprocess_folders[0])
        sorted_unique_indices = get_sorted_unique_indices(voxels_path)

        if not sorted_unique_indices:
            print(f"源文件夹 {source_root} 中没有有效的pkl文件，跳过处理")
            continue

        if idx == 0:
            train_indices = sorted_unique_indices
            test_indices = []
        elif idx == 1:
            train_indices = []
            test_indices = sorted_unique_indices


        # 遍历当前源文件夹下的子文件夹
        for folder_name in os.listdir(source_root):
            if folder_name not in preprocess_folders:
                continue
            folder_path = os.path.join(source_root, folder_name)

            if not os.path.isdir(folder_path):
                continue

            # 处理当前文件夹中的文件
            for filename in os.listdir(folder_path):
                if not filename.endswith(".pkl"):
                    continue

                try:
                    # 提取文件名中的数字部分
                    name_part = os.path.splitext(filename)[0]
                    file_num = int(name_part)

                    # 构建相对路径（如：20250711_1519_finterval1/voxels/000600.pkl）
                    relative_path = f"{source_name}/{folder_name}/{filename}"

                    # 根据索引列表划分训练集和测试集
                    if file_num in train_indices:
                        train_files.append(relative_path)
                    elif file_num in test_indices:
                        test_files.append(relative_path)

                except ValueError:
                    continue

    # 保存训练集文件路径
    with open(os.path.join(output_dir, f'train_files_split_completed_scenes.txt'), 'w') as f:
        for path in train_files:
            f.write(f"{path}\n")

    # 保存测试集文件路径
    with open(os.path.join(output_dir, f'test_files_split_completed_scenes.txt'), 'w') as f:
        for path in test_files:
            f.write(f"{path}\n")





if __name__ == "__main__":
    # 执行处理
    process_honor_files()
    print("honor_coll_data文件处理完成！")