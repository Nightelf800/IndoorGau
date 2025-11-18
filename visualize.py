import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from collections import Counter
from rich.progress import track
from visual.syndata_visualize import draw

from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks, build_from_configs, evaluation

class_names = ['empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa',
               'table', 'tvs', 'furn', 'objs', 'people']
NYU_COLORS = np.array(
    [
        [22, 191, 206, 255],  # 0 empty    浅青色
        [214, 38, 40, 255],  # 1 ceiling   红色
        [43, 160, 43, 255],  # 2 floor     绿色
        [158, 216, 229, 255],  # 3 wall    淡青色
        [114, 158, 206, 255],  # 4 window   灰蓝色
        [204, 204, 91, 255],  # 5 chair     浅黄色
        [255, 186, 119, 255],  # 6 bed      橙色
        [147, 102, 188, 255],  # 7 sofa     紫色
        [30, 119, 181, 255],  # 8 table     深蓝色
        [188, 188, 33, 255],  # 9 tvs       亮黄色
        [255, 127, 12, 255],  # 10 furn     橙色
        [196, 175, 214, 255],  # 11 objs       淡紫色
        [153, 153, 153, 255],  # 12     灰色
        [219, 68, 55, 255],  # 12 people    深粉色
    ]
)


def log_metrics(evaluator, prefix=None, scene=False):
    metrics = evaluator.compute()
    iou_per_class = metrics.pop('iou_per_class')
    if prefix:
        metrics = {'/'.join((prefix, k)): v.item() for k, v in metrics.items()}
    if scene:
        print(f'*' * 10, 'scene', '*' * 10)
    else:
        print(f'*' * 20)
    print(f'metrics: {metrics}')
    print(f'*' * 20)
    evaluator.reset()


def get_grid_coords(dims, resolution, vox_origin):
    """
    计算体素网格中每个体素的中心坐标
    :param dims: 体素网格的尺寸 [x, y, z]
    :param resolution: 体素分辨率
    :param vox_origin: 体素网格在世界坐标系中的原点
    :return: 体素中心坐标数组，形状为 (N, 3)
    """
    g_xx = np.arange(0, dims[0])
    g_yy = np.arange(0, dims[1])
    g_zz = np.arange(0, dims[2])

    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz, indexing='ij')
    # coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    # coords_grid = coords_grid.astype(float)
    coords_grid = np.concatenate([xx.reshape(
        1, -1), yy.reshape(1, -1), zz.reshape(1, -1)], axis=0).astype(int).T
    coords_grid = (coords_grid * resolution) + resolution / 2
    # 若 vox_origin 是 PyTorch 张量，转换为 NumPy 数组
    if isinstance(vox_origin, torch.Tensor):
        vox_origin = vox_origin.cpu().numpy()
    coords_grid += vox_origin
    return coords_grid


def get_grid_coords_v2(dims, resolution, vox_origin):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """
    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(float)
    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)
    if isinstance(vox_origin, torch.Tensor):
        vox_origin = vox_origin.cpu().numpy()
    coords_grid += vox_origin
    return coords_grid


def remove_voxel_redundancy(points, voxel_size=0.08):
    """
    去除体素冗余点，每个体素保留一个点

    参数:
        points: 形状为(N, 4)的numpy数组，前3列为xyz坐标，第4列为类别
        voxel_size: 体素大小，默认0.08

    返回:
        形状为(N_NEW, 4)的numpy数组，每个体素只保留一个点
    """
    # 提取坐标信息
    coords = points[:, :3]

    # 计算每个点对应的体素索引（将连续坐标离散化为体素网格索引）
    # 这里使用floor除法并取整，将相同体素的点映射到同一索引
    voxel_indices = (coords / voxel_size).astype(int)

    # 将三维索引转换为字符串键，用于识别相同体素
    # 例如将(x_idx, y_idx, z_idx)转换为"x,y,z"格式的字符串
    voxel_keys = [f"{x},{y},{z}" for x, y, z in voxel_indices]

    # 使用字典保存每个体素的第一个出现的点
    unique_voxels = {}
    for key, point in zip(voxel_keys, points):
        if key not in unique_voxels:
            # 体素不存在时直接添加
            unique_voxels[key] = point
        else:
            # 体素已存在时，检查现有类是否为0或255
            existing_cls = unique_voxels[key][3]
            # 如果现有类别不是0也不是255，则跳过当前点（不更新）
            if existing_cls not in (0, 255):
                continue
            # 否则（现有类别是0或255），用当前点更新
            unique_voxels[key] = point

    # 将字典的值转换为numpy数组，得到去重后的结果
    result = np.array(list(unique_voxels.values()))

    return result


def merge_voxels_to_world(voxels, target_shape=None, voxel_size=0.08, vox_origins=None, label=False):
    """
    将多帧体素数据映射到世界坐标系并重新构造体素网格
    :param voxels: 多帧体素数据，形状为 [30, 1, 12, 60, 60, 36]
    :param voxel_size: 体素分辨率，单位：米
    :param vox_origins: 每帧体素网格在世界坐标系中的原点，形状为 [30, 3]
    :return: 重新构造后的体素网格
    """
    num_frames = voxels.shape[0]
    if vox_origins is None:
        vox_origins = np.zeros((num_frames, 3))

    all_coords = []

    for i in range(num_frames):
        frame_voxels = voxels[i]

        if not label:
            frame_voxels = torch.softmax(frame_voxels, dim=0).detach().cpu().numpy()
            pred_classes = np.argmax(frame_voxels, axis=0)
        else:
            pred_classes = frame_voxels.detach().cpu().numpy()

        grid_coords = get_grid_coords_v2([pred_classes.shape[0], pred_classes.shape[1], pred_classes.shape[2]],
                                         voxel_size, vox_origins[i])
        coords = np.vstack([grid_coords.T, pred_classes.reshape(-1)]).T
        # print(f'pred_classes.shape: {pred_classes.shape}')
        all_coords.append(coords)

    all_coords = np.vstack(all_coords)
    # print(f'all_coords.shape: {all_coords.shape}')

    all_coords_filtered = remove_voxel_redundancy(all_coords)
    # print(f'all_coords_filtered.shape: {all_coords_filtered.shape}')
    # draw_scene_test(all_coords_filtered, voxel_size=0.08, colors=NYU_COLORS)

    return all_coords_filtered


def points_to_voxel_grid(points, voxel_size=0.08, target_shape=None):
    """
    将去重后的点转换为N×M×H的体素网格

    参数:
        points: 形状为(N_NEW, 4)的numpy数组，前3列为xyz坐标，第4列为类别
        voxel_size: 体素大小，默认0.08

    返回:
        voxel_grid: 形状为(N, M, H)的numpy数组，每个元素表示体素的类别
        min_coords: 体素网格原点坐标(最小x, 最小y, 最小z)，用于坐标映射
    """
    # 提取坐标和类别
    coords = points[:, :3]
    classes = points[:, 3].astype(int)

    # 计算坐标的最小值（作为体素网格的原点）
    min_coords = np.min(coords, axis=0)  # (x_min, y_min, z_min)

    # 计算每个点在体素网格中的索引
    # 公式：体素索引 = (坐标 - 最小坐标) / 体素大小 → 向下取整
    voxel_indices = ((coords - min_coords) / voxel_size).astype(int)

    # 处理目标形状
    if target_shape is not None:
        # 确保目标形状是三维的
        assert len(target_shape) == 3, "target_shape必须是三维元组(N, M, H)"
        # 过滤超出目标形状范围的索引（只保留有效索引的点）
        valid_mask = (
                (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < target_shape[0]) &
                (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < target_shape[1]) &
                (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < target_shape[2])
        )
        # 筛选有效点
        voxel_indices = voxel_indices[valid_mask]
        classes = classes[valid_mask]
        # 使用指定的目标形状
        grid_dims = target_shape
    else:
        # 自动计算网格尺寸
        grid_dims = np.max(voxel_indices, axis=0) + 1  # (N, M, H)

    # 初始化体素网格（用0表示空体素，可根据需要修改）
    voxel_grid = np.zeros(grid_dims, dtype=int)

    # 填充体素网格：将每个点的类别写入对应的体素位置
    for idx, cls in zip(voxel_indices, classes):
        x, y, z = idx
        if voxel_grid[x, y, z] in (0, 255):
            voxel_grid[x, y, z] = cls  # 若有重复索引（理论上不应存在），后出现的会覆盖前一个

    return voxel_grid, min_coords


def transform_voxel_grid_to_new_origin(prev_voxel_grid, prev_origin, current_origin, voxel_size=0.08):
    """
    将上一帧的体素网格转换到当前帧的体素原点坐标系中
    :param prev_voxel_grid: 上一帧的体素网格，形状为 (D, H, W)
    :param prev_origin: 上一帧的体素原点，形状为 (3,)
    :param current_origin: 当前帧的体素原点，形状为 (3,)
    :param voxel_size: 体素大小，默认0.08
    :return: 转换后的体素网格，形状为 (D, H, W)
    """
    # 确保输入是numpy数组
    if isinstance(prev_voxel_grid, torch.Tensor):
        prev_voxel_grid = prev_voxel_grid.detach().cpu().numpy()
    if isinstance(prev_origin, torch.Tensor):
        prev_origin = prev_origin.detach().cpu().numpy()
    if isinstance(current_origin, torch.Tensor):
        current_origin = current_origin.detach().cpu().numpy()

    # 获取上一帧体素网格的尺寸
    grid_dims = prev_voxel_grid.shape

    # 计算每个体素在世界坐标系中的中心坐标
    grid_coords = get_grid_coords_v2(grid_dims, voxel_size, prev_origin)

    # 创建一个新的体素网格用于存储转换后的结果
    transformed_voxel_grid = np.zeros_like(prev_voxel_grid)

    # 将每个体素从世界坐标系转换到当前帧的体素坐标系
    for i in range(grid_coords.shape[0]):
        x, y, z = grid_coords[i]
        # 计算在当前帧体素网格中的索引
        current_x = int((x - current_origin[0]) / voxel_size)
        current_y = int((y - current_origin[1]) / voxel_size)
        current_z = int((z - current_origin[2]) / voxel_size)

        # 检查索引是否在当前帧体素网格的范围内
        if (0 <= current_x < grid_dims[0] and
                0 <= current_y < grid_dims[1] and
                0 <= current_z < grid_dims[2]):
            # 获取上一帧中对应位置的体素值
            prev_x = int((x - prev_origin[0]) / voxel_size)
            prev_y = int((y - prev_origin[1]) / voxel_size)
            prev_z = int((z - prev_origin[2]) / voxel_size)

            # 将上一帧的体素值复制到当前帧的对应位置
            if (0 <= prev_x < grid_dims[0] and
                    0 <= prev_y < grid_dims[1] and
                    0 <= prev_z < grid_dims[2]):
                transformed_voxel_grid[current_x, current_y, current_z] = prev_voxel_grid[prev_x, prev_y, prev_z]

    return transformed_voxel_grid


@hydra.main(config_path='configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    cfg, _ = pre_build_callbacks(cfg)

    dls, meta_info = build_data_loaders(cfg.data)
    data_loader = dls[1]

    if cfg.get('ckpt_path'):
        model = LitModule.load_from_checkpoint(cfg.ckpt_path, **cfg, meta_info=meta_info)
    else:
        import warnings
        warnings.warn('\033[31;1m{}\033[0m'.format('No ckpt_path is provided'))
        model = LitModule(**cfg, meta_info=meta_info)

    test_evaluator = build_from_configs(evaluation, cfg.evaluator).cuda()
    # test_scene_evaluator = build_from_configs(evaluation, cfg.evaluator2).cuda()
    model.cuda()
    model.eval()
    total_steps = len(data_loader)
    total_time = 0.0

    with torch.no_grad():
        for batch_inputs, targets in track(data_loader):
            # print(batch_inputs.keys())
            # print('batch_inputs.name: {}'.format(batch_inputs['name']))
            scenes = batch_inputs['scene']  # 假设 'name' 标识场景
            filenames = batch_inputs['filename']

            targets = {key: targets[key].cuda() for key in targets}
            # tar = targets['target']
            # mask = torch.where((tar != 0) & (tar != 255))
            # tar = tar[mask]
            # for i in range(len(class_names)):
            #     print('class: {}, occ: {}'.format(class_names[i], torch.sum(tar==i)/tar.size(0)))

            for key in batch_inputs:
                if isinstance(batch_inputs[key], torch.Tensor):
                    batch_inputs[key] = batch_inputs[key].cuda()

            start_time = time.time()  # 开始计时
            outputs = model(batch_inputs)
            step_time = time.time() - start_time  # 计算每步所用的时间
            if test_evaluator:
                test_evaluator.update(outputs, targets)

            batch_size = len(scenes)
            assert batch_size == 1
            for i in range(batch_size):
                scene = scenes[i]
                filename = filenames[i]
                voxel_origin = batch_inputs['voxel_origin'][i].detach().cpu().numpy()
                cam_pose = batch_inputs['cam_pose'][i].detach().cpu().numpy()
                fov_mask = batch_inputs['fov_mask_1'][i].detach().cpu().numpy()
                cam_K = batch_inputs['cam_K'][i].detach().cpu().numpy()


                prev_occ_output = torch.softmax(outputs['ssc_logits'][i], dim=0).argmax(dim=0).detach().cpu().numpy()
                prev_voxel_origin = voxel_origin
                target_np = targets['target'][i].detach().cpu().numpy()

                print(f'############## visual ##############')
                print(f'pred visual')
                print(f'file: {filename}')
                print(f'scene: {scene}')
                draw('SYNData', prev_occ_output, cam_pose, prev_voxel_origin, fov_mask,
                     (640, 480), cam_K[0, 0], need_update_view=False)
                # draw('SYNData', prev_occ_output, cam_pose, prev_voxel_origin, fov_mask,
                #      (640, 480), cam_K[0, 0], save_path='./outputs/visual/1014/pred/', file_name=f'visual_{scene}_{filename}.png',
                #      need_update_view=False)
                print(f'target visual')
                draw('SYNData', target_np, cam_pose, prev_voxel_origin, fov_mask,
                     (640, 480), cam_K[0, 0], need_update_view=False)
                # draw('SYNData', target_np, cam_pose, prev_voxel_origin, fov_mask,
                #      (640, 480), cam_K[0, 0], save_path='./outputs/visual/1014/gt/', file_name=f'visual_{scene}_{filename}.png',
                #      need_update_view=False)


            fps = 1 / step_time  # 计算FPS
            total_time += step_time

            # print(f"FPS: {fps:.2f}")

        log_metrics(test_evaluator, 'val')

        average_fps = total_steps / total_time  # 计算平均FPS
        print(f"Average FPS over {total_steps} steps: {average_fps:.2f}")


if __name__ == '__main__':
    main()
