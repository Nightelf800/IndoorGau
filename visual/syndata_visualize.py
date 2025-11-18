import os
import pickle
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from rich.progress import track
from mayavi import mlab
from threading import Thread

from scipy.ndimage import zoom

from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks, build_from_configs, evaluation
from ssc_pl.utils.helper import vox2pix

from xvfbwrapper import Xvfb

COLORS = np.array([
    [100, 150, 245, 255],
    [100, 230, 245, 255],
    [30, 60, 150, 255],
    [80, 30, 180, 255],
    [100, 80, 250, 255],
    [255, 30, 30, 255],
    [255, 40, 200, 255],
    [150, 30, 90, 255],
    [255, 0, 255, 255],
    [255, 150, 255, 255],
    [75, 0, 75, 255],
    [175, 0, 75, 255],
    [255, 200, 0, 255],
    [255, 120, 50, 255],
    [0, 175, 0, 255],
    [135, 60, 0, 255],
    [150, 240, 80, 255],
    [255, 240, 150, 255],
    [255, 0, 0, 255],
]).astype(np.uint8)

KITTI360_COLORS = np.concatenate((
    COLORS[0:6],
    COLORS[8:15],
    COLORS[16:],
    np.array([[250, 150, 0, 255], [50, 255, 255, 255]]).astype(np.uint8),
), 0)

NYU_COLORS = np.array(
    [
        [22, 191, 206, 255],  # 0 empty    浅青色
        [214, 38, 40, 255],  # 1 ceiling   红色
        [43, 160, 43, 255],  # 2 floor     绿色
        [158, 216, 229, 255],  # 3 wall    淡青色
        [114, 158, 206, 255],  # 4 window   灰蓝色
        [204, 204, 91, 255],  # 5 chair     浅黄色
        [30, 119, 181, 255],  # 6 table     深蓝色
        [188, 188, 33, 255],  # 7 tvs       亮黄色
        [255, 127, 12, 255],  # 8 furn     橙色
        [196, 175, 214, 255],  # 9 objs       淡紫色
        [153, 153, 153, 255],  # 10 people     灰色
        [219, 68, 55, 255],  # 11     深粉色
    ]
)

COLORS_MAPS = ['empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'table', 'tvs', 'furn', 'objs', 'people']


def gt_visual_cal_fov_mask(outputs):
    cam_pose = np.linalg.inv(outputs['cam_pose'])
    voxel_origin = outputs['voxel_origin']
    cam_K = np.array(((518.8579, 0, 320), (0, 518.8579, 240), (0, 0, 1)))
    voxel_size = 0.04  # meters
    scene_size = (4, 4, 2)  # meters
    img_shape = (640, 480)

    projected_pix, fov_mask, pix_z = vox2pix(cam_pose, cam_K, voxel_origin,
                                             voxel_size, img_shape, scene_size)
    return fov_mask


def get_grid_coords(dims, resolution):
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
    return coords_grid

def draw_scene(
        voxels,
        cam_pose,
        vox_origin,
        fov_mask,
        img_size,
        f,
        voxel_size=0.08,
        d=0.5,  # 7m - determine the size of the mesh representing the camera
        colors=NYU_COLORS,
        save_path=None,
        file_name=None,
        need_update_view=False,
):
    # Compute the coordinates of the mesh representing camera
    x = d * img_size[0] / (2 * f)
    y = d * img_size[1] / (2 * f)
    tri_points = np.array([
        [0, 0, 0],
        [x, y, d],
        [-x, y, d],
        [-x, -y, d],
        [x, -y, d],
    ])
    tri_points = np.hstack([tri_points, np.ones((5, 1))])
    tri_points = (np.linalg.inv(cam_pose) @ tri_points.T).T
    x = tri_points[:, 0] - vox_origin[0]
    y = tri_points[:, 1] - vox_origin[1]
    z = tri_points[:, 2] - vox_origin[2]
    triangles = [
        (0, 1, 2),
        (0, 1, 4),
        (0, 3, 4),
        (0, 2, 3),
    ]

    # Compute the voxels coordinates
    print(f'voxel_size: {voxel_size}')
    grid_coords = get_grid_coords([voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size)
    print(f'grid_coords.shape: {grid_coords.shape}')
    # Attach the predicted class to every voxel
    print(f'voxels.reshape(-1).shape: {voxels.reshape(-1).shape}')
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
    # fov_grid_coords = grid_coords[fov_mask, :]
    # Get the voxels outside FOV
    # outfov_grid_coords = grid_coords[~fov_mask, :]
    # Draw the camera
    # mlab.options.offscreen = True
    try:
        fig = mlab.figure(bgcolor=(1, 1, 1))  # 使用指定的场景
        print("图形窗口创建成功")
    except Exception as e:
        print(f"创建图形窗口时出错: {e}")
    mlab.triangular_mesh(
        x, y, z, triangles, representation='wireframe', color=(0, 0, 0), line_width=5)

    outfov_colors = colors.copy()
    outfov_colors[:, :3] = outfov_colors[:, :3] // 3 * 2


    voxels = grid_coords[(grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)]

    plt_plot = mlab.points3d(
        voxels[:, 0],
        voxels[:, 1],
        voxels[:, 2],
        voxels[:, 3],
        colormap='viridis',
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode='cube',
        opacity=1.0,
        vmin=0,
        vmax=10)

    plt_plot.glyph.scale_mode = 'scale_by_vector'
    plt_plot.module_manager.scalar_lut_manager.lut.table = colors
    # print(f'save: {save}')
    # 在颜色条旁边列出对应的物体名称
    for i, name in enumerate(COLORS_MAPS):
        color = tuple(NYU_COLORS[i][:3] / 255.0)
        mlab.text(0.9, 0.4 * (1 - (i + 0.5) / len(COLORS_MAPS)), name, color=color, width=len(name)*0.02)
    plt_plot.scene.camera.zoom(1.0)
    # 调整相机视角（可选，如果你想通过代码设置视角）

    if need_update_view:
        def update_view():
            azimuth = 160
            elevation = -30
            roll = 90
            while True:
                time.sleep(3)  # 每 2 秒更新一次视角
                azimuth += 10  # 每次增加 10 度
                # elevation += 20
                # roll += 10

                mlab.view(azimuth=azimuth, elevation=elevation, roll=roll, distance=7)
                print(f'azimuth: {azimuth}, elevation: {elevation}, roll: {roll}')
                mlab.gcf().scene.render()  # 强制重绘场景

        # 启动一个线程来更新视角
        update_thread = Thread(target=update_view)
        update_thread.daemon = True
        update_thread.start()
        # 显示场景
        mlab.show()
    else:
        mlab.view(azimuth=190, elevation=-30, roll=90, distance=7)

        if save_path:
            if not os.path.exists(save_path):
            # 如果不存在，创建路径
                os.makedirs(save_path)
            save = os.path.join(save_path, file_name)
            print(f'save: {save}')
            mlab.savefig(save, size=(1920, 1080))
            mlab.close()
        else:
            mlab.show()

def draw(
        data_type,
        voxels,
        cam_pose,
        vox_origin,
        fov_mask,
        img_size,
        f,
        voxel_size=0.08,
        d=0.5,  # 7m - determine the size of the mesh representing the camera
        colors=NYU_COLORS,
        save_path=None,
        file_name=None,
        need_update_view=False,
):
    # Compute the coordinates of the mesh representing camera
    x = d * img_size[0] / (2 * f)
    y = d * img_size[1] / (2 * f)
    tri_points = np.array([
        [0, 0, 0],
        [x, y, d],
        [-x, y, d],
        [-x, -y, d],
        [x, -y, d],
    ])
    tri_points = np.hstack([tri_points, np.ones((5, 1))])
    tri_points = (np.linalg.inv(cam_pose) @ tri_points.T).T
    x = tri_points[:, 0] - vox_origin[0]
    y = tri_points[:, 1] - vox_origin[1]
    z = tri_points[:, 2] - vox_origin[2]
    triangles = [
        (0, 1, 2),
        (0, 1, 4),
        (0, 3, 4),
        (0, 2, 3),
    ]

    # Compute the voxels coordinates
    print(f'voxel_size: {voxel_size}')
    grid_coords = get_grid_coords([voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size)
    # print(f'grid_coords.shape: {grid_coords.shape}')
    # Attach the predicted class to every voxel
    # print(f'voxels.reshape(-1).shape: {voxels.reshape(-1).shape}')
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords[fov_mask, :]
    # Get the voxels outside FOV
    outfov_grid_coords = grid_coords[~fov_mask, :]
    # Draw the camera
    mlab.options.offscreen = True if save_path else False
    try:
        fig = mlab.figure(bgcolor=(1, 1, 1))  # 使用指定的场景
        print("图形窗口创建成功")
    except Exception as e:
        print(f"创建图形窗口时出错: {e}")
    mlab.triangular_mesh(
        x, y, z, triangles, representation='wireframe', color=(0, 0, 0), line_width=5)

    outfov_colors = colors.copy()
    outfov_colors[:, :3] = outfov_colors[:, :3] // 3 * 2

    for i, grid_coords in enumerate((fov_grid_coords, outfov_grid_coords)):
        # Remove empty and unknown voxels
        voxels = grid_coords[(grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)]

        plt_plot = mlab.points3d(
            voxels[:, 0],
            voxels[:, 1],
            voxels[:, 2],
            voxels[:, 3],
            colormap='viridis',
            scale_factor=voxel_size - 0.05 * voxel_size,
            mode='cube',
            opacity=1.0,
            vmin=0,
            vmax=12)

        plt_plot.glyph.scale_mode = 'scale_by_vector'
        plt_plot.module_manager.scalar_lut_manager.lut.table = colors if i == 0 else outfov_colors
    # print(f'save: {save}')
    # 在颜色条旁边列出对应的物体名称
    for i, name in enumerate(COLORS_MAPS):
        color = tuple(NYU_COLORS[i][:3] / 255.0)
        mlab.text(0.9, 0.4 * (1 - (i + 0.5) / len(COLORS_MAPS)), name, color=color, width=len(name)*0.02)
    plt_plot.scene.camera.zoom(1.0)
    # 调整相机视角（可选，如果你想通过代码设置视角）

    if need_update_view:
        def update_view():
            azimuth = 0
            elevation = 60
            roll = 0
            while True:
                time.sleep(4)  # 每 2 秒更新一次视角
                azimuth += 10  # 每次增加 10 度
                # elevation += 15
                # roll += 15

                mlab.view(azimuth=azimuth, elevation=elevation, distance=8)
                print(f'azimuth: {azimuth}, elevation: {elevation}, roll: {roll}')
                mlab.gcf().scene.render()  # 强制重绘场景


        # 启动一个线程来更新视角
        update_thread = Thread(target=update_view)
        update_thread.daemon = True
        update_thread.start()
        # 显示场景
        mlab.show()
    else:
        mlab.view(azimuth=210, elevation=60, distance=10)

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            save = os.path.join(save_path, file_name)
            print(f'save: {save}')
            mlab.savefig(save)
            mlab.close()
        else:
            mlab.show()

def draw_gt(
        voxels,
        cam_pose,
        vox_origin,
        img_size,
        f,
        voxel_size=0.08,
        d=0.5,  # 7m - determine the size of the mesh representing the camera
        colors=NYU_COLORS,
        save_path=None,
        file_name=None,
        need_update_view=False,
):
    # Compute the coordinates of the mesh representing camera
    x = d * img_size[0] / (2 * f)
    y = d * img_size[1] / (2 * f)
    tri_points = np.array([
        [0, 0, 0],
        [x, y, d],
        [-x, y, d],
        [-x, -y, d],
        [x, -y, d],
    ])
    tri_points = np.hstack([tri_points, np.ones((5, 1))])
    tri_points = (np.linalg.inv(cam_pose) @ tri_points.T).T
    x = tri_points[:, 0] - vox_origin[0]
    y = tri_points[:, 1] - vox_origin[1]
    z = tri_points[:, 2] - vox_origin[2]
    triangles = [
        (0, 1, 2),
        (0, 1, 4),
        (0, 3, 4),
        (0, 2, 3),
    ]

    # Compute the voxels coordinates
    print(f'voxel_size: {voxel_size}')
    grid_coords = get_grid_coords([voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size)
    # print(f'grid_coords.shape: {grid_coords.shape}')
    # Attach the predicted class to every voxel
    # print(f'voxels.reshape(-1).shape: {voxels.reshape(-1).shape}')
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Draw the camera
    mlab.options.offscreen = True if save_path else False
    try:
        fig = mlab.figure(bgcolor=(1, 1, 1))  # 使用指定的场景
        print("图形窗口创建成功")
    except Exception as e:
        print(f"创建图形窗口时出错: {e}")
    mlab.triangular_mesh(
        x, y, z, triangles, representation='wireframe', color=(0, 0, 0), line_width=5)

    outfov_colors = colors.copy()
    outfov_colors[:, :3] = outfov_colors[:, :3] // 3 * 2

    voxels = grid_coords[(grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)]

    plt_plot = mlab.points3d(
        voxels[:, 0],
        voxels[:, 1],
        voxels[:, 2],
        voxels[:, 3],
        colormap='viridis',
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode='cube',
        opacity=1.0,
        vmin=0,
        vmax=12)

    plt_plot.glyph.scale_mode = 'scale_by_vector'
    plt_plot.module_manager.scalar_lut_manager.lut.table = colors
    # print(f'save: {save}')
    # 在颜色条旁边列出对应的物体名称
    for i, name in enumerate(COLORS_MAPS):
        color = tuple(NYU_COLORS[i][:3] / 255.0)
        mlab.text(0.9, 0.4 * (1 - (i + 0.5) / len(COLORS_MAPS)), name, color=color, width=len(name)*0.02)
    plt_plot.scene.camera.zoom(1.0)
    # 调整相机视角（可选，如果你想通过代码设置视角）

    if need_update_view:
        def update_view():
            azimuth = 0
            elevation = 0
            roll = 0
            while True:
                time.sleep(3)  # 每 2 秒更新一次视角
                azimuth += 10  # 每次增加 10 度
                # elevation += 20
                # roll += 10

                mlab.view(azimuth=azimuth, elevation=60, distance=7)
                print(f'azimuth: {azimuth}, elevation: {elevation}, roll: {roll}')
                mlab.gcf().scene.render()  # 强制重绘场景


        # 启动一个线程来更新视角
        update_thread = Thread(target=update_view)
        update_thread.daemon = True
        update_thread.start()
        # 显示场景
        mlab.show()
    else:
        mlab.view(azimuth=-90, elevation=60, distance=7)

        if save_path:
            if not os.path.exists(save_path):
                # 如果不存在，创建路径
                os.makedirs(save_path)
            save = os.path.join(save_path, file_name)
            print(f'save: {save}')
            mlab.savefig(save)
            # mlab.savefig(save, size=(1920, 1080))
            mlab.close()
        else:
            mlab.show()


def log_metrics(evaluator, prefix=None):
    metrics = evaluator.compute()
    iou_per_class = metrics.pop('iou_per_class')
    print(f'metrics: {metrics}')
    if prefix:
        metrics = {'/'.join((prefix, k)): v.item() for k, v in metrics.items()}
    print(f'metrics: {metrics}')
    evaluator.reset()


@hydra.main(config_path='configs', config_name='config', version_base=None)
def main(config: DictConfig):
    files = ([os.path.join(config.path, f)
              for f in os.listdir(config.path)] if os.path.isdir(config.path) else [config.path])
    test_evaluator = build_from_configs(evaluation, config.evaluator).cuda()

    print(f'----------files------------')
    print(f'files: {files}')

    for file in track(files):
        with open(file, 'rb') as f:
            outputs = pickle.load(f)
        id_name = file.split('/')[-1].split('.')[0]
        print(f'id_name: {id_name}')
        print(f'outputs.keys: {outputs.keys()}')

        cam_K = outputs['cam_K']
        cam_pose = outputs['cam_pose'] if 'cam_pose' in outputs else outputs[
            'T_velo_2_cam']  # compatible with MonoScene
        vox_origin = outputs['voxel_origin'] if 'voxel_origin' in outputs else np.array([0, -25.6, -2])
        fov_mask = outputs['fov_mask_1'] if 'fov_mask_1' in outputs else gt_visual_cal_fov_mask(outputs)
        pred = outputs['pred'] if 'pred' in outputs else None
        preds_ori = outputs['preds_ori'] if 'preds_ori' in outputs else None
        target = outputs['target'] if 'target' in outputs else outputs.pop('target_1_4').transpose(0, 2, 1)
        # if test_evaluator:
        #     test_evaluator.update({'ssc_logits': torch.tensor(preds_ori.astype(np.int32)).unsqueeze(0).cuda()},
        #                           {'target': torch.tensor(target.astype(np.int32)).unsqueeze(0).cuda()})
        # log_metrics(test_evaluator, 'val')

        # 计算缩放因子
        # 使用zoom函数进行下采样
        # zoom_factors = (50 / 60, 50 / 60, 25 / 36)
        # target = zoom(target, zoom_factors, order=0)  # order=3 是三次插值
        # 直接下采样
        # target = target[::2, ::2, ::2]

        # print(f'------------output-------------')
        # print(f'output.keys: {outputs.keys()}')
        # print('output[pred].shape: {}'.format(outputs['pred'].shape))
        # print('output[target].shape: {}'.format(outputs['target'].shape))

        data_type = config.data.datasets.type
        if data_type == 'SemanticKITTI':
            params = dict(
                img_size=(1220, 370),
                f=707.0912,
                voxel_size=0.2,
                d=7,
                colors=COLORS,
            )
        elif data_type == 'KITTI360':
            # Otherwise the trained model would output distorted results, due to unreasonably labeling
            # a large number of voxels as "ignored" in the annotations.
            pred[target == 255] = 0
            params = dict(
                img_size=(1408, 376),
                f=552.55426,
                voxel_size=0.2,
                d=7,
                colors=KITTI360_COLORS,
            )
        elif data_type == 'NYUv2':
            pred[target == 255] = 0
            params = dict(
                img_size=(640, 480),
                f=518.8579,
                voxel_size=0.04,
                d=0.75,
                colors=NYU_COLORS,
            )
        elif data_type == 'SYNData':
            pred[target == 255] = 0
            params = dict(
                img_size=(640, 480),
                f=cam_K[0, 0],
                voxel_size=0.08,
                d=0.5,
                colors=NYU_COLORS,
            )
        elif data_type == 'ScanNet':
            pred[target == 255] = 0
            params = dict(
                img_size=(640, 480),
                f=116.9621,
                voxel_size=0.08,
                d=0.3,
                colors=NYU_COLORS,
            )
        else:
            raise NotImplementedError

        print(f'config.data.datasets.type: {config.data.datasets.type}')
        print(f'vox_origin: {vox_origin}')
        file_name = file.split(os.sep)[-1].split(".")[0]
        for i, vol in enumerate((pred, target)):
            if vol is None:
                continue
            # print(f'{i} | vol: {vol.shape}')
            # draw(vol, cam_pose, vox_origin, fov_mask, **params,
            #      save_path=f'./outputs/visual', file_name=f'{id_name}_{i}.png')
            draw(data_type, vol, cam_pose, vox_origin, fov_mask, **params, need_update_view=False)


if __name__ == '__main__':
    main()
