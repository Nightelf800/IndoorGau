import numpy as np

def extract_intrinsic_matrix(line):
    """
    从 COLMAP 的 cameras.txt 文件的一行中提取 3x3 相机内参矩阵。

    参数:
    line (str): cameras.txt 文件中的一行，格式如 "1 PINHOLE 640 480 888.8890923394097 1000.0002288818359 320.0 240.0"

    返回:
    list: 3x3 相机内参矩阵
    """
    parts = line.split()
    # 提取内参
    fx = float(parts[4])
    fy = float(parts[5])
    cx = float(parts[6])
    cy = float(parts[7])

    # 构建 3x3 内参矩阵
    K = [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ]
    return K

def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为 3x3 旋转矩阵。

    参数:
    q (list): 四元数 [qw, qx, qy, qz]

    返回:
    np.ndarray: 3x3 旋转矩阵
    """
    w, x, y, z = q
    R = np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
    ])
    return R


def extract_extrinsic_matrix(line):
    """
    从 COLMAP 的 images.txt 文件的一行中提取 4x4 相机外参矩阵。

    参数:
    line (str): images.txt 文件中的一行，格式如 "1 -6.5 1.0 1.2 0.789969203269606 -89.95437359162356 -89.21003054627484 1 observation_rgb_1.png"

    返回:
    np.ndarray: 4x4 相机外参矩阵
    """
    parts = line.split()
    # 提取四元数
    qw = float(parts[1])
    qx = float(parts[2])
    qy = float(parts[3])
    qz = float(parts[4])
    # 提取平移向量
    tx = float(parts[5])
    ty = float(parts[6])
    tz = float(parts[7])

    # 将四元数转换为旋转矩阵
    R = quaternion_to_rotation_matrix([qw, qx, qy, qz])
    # 构建 4x4 外参矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T

def read_cameras_file(file_path):
    """
    读取 COLMAP 的 cameras.txt 文件，并提取每行的 3x3 相机内参矩阵。

    参数:
    file_path (str): cameras.txt 文件的路径

    返回:
    list: 包含每个相机内参矩阵的列表
    """
    intrinsic_matrices = []
    extrinsic_matrices = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                matrix = extract_intrinsic_matrix(line)
                intrinsic_matrices.append(matrix)
                matrix = extract_extrinsic_matrix(line)
                extrinsic_matrices.append(matrix)
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except Exception as e:
        print(f"读取文件时出错: {e}")
    return intrinsic_matrices


if __name__ == "__main__":
    file_path = "/mnt/bn/yuanlichen0610modeleval/datasets/syndata/v2_straight/render_gls20250713/sparse/0/cameras.txt"
    matrices = read_cameras_file(file_path)
    for i, matrix in enumerate(matrices):
        print(f"相机 {i + 1} 的内参矩阵:")
        for row in matrix:
            print(row)
        print()