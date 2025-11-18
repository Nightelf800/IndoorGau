import json
import os.path as osp
import glob

import cv2
import numpy as np
import torch
import pickle

from PIL import Image
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from ...utils.helper import vox2pix, compute_local_frustums, compute_CP_mega_matrix, get_meshgrid
from depth_eval.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from ...engine import LitModule
from cfg_module import ConfigManager
from ...utils.fusion import TSDFVolume
from torchvision.transforms.functional import InterpolationMode

BICUBIC = InterpolationMode.BICUBIC


# from featup.train_jbu_upsampler import JBUFeatUp

# ckpt_path = '/share/lkl/Symphonies/outputs/11_19_dim64_sym/e25_miou0.2860.ckpt'
class SYNDataScenes11Class(Dataset):
    META_INFO = {
        'class_weights':
            torch.tensor((0.05, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
        'class_names': ('empty', 'ceiling', 'floor', 'wall', 'window', 'chair',
                        'table', 'tvs', 'furn', 'objs', 'people'),
    }

    def __init__(self, split, data_root, label_root, voxel_size=0.08, num_classes=12, pc_range=None, depth_root=None,
                 use_crop=True, use_depth_eval=False, frustum_size=4, depth_encoder='null', use_tsdf=False):
        self.data_root = data_root
        self.label_root = data_root
        self.depth_root = data_root
        self.split = split
        self.frustum_size = frustum_size
        self.num_classes = num_classes
        print(f'num_classes: {self.num_classes}')
        self.use_tsdf = use_tsdf
        self.voxel_size = voxel_size  # meters
        self.use_crop = use_crop  # crop or scale
        self.use_depth_eval = use_depth_eval

        self.scene_size = (4, 4, 2)  # meters
        # self.scene_size = (4, 4, 2)  # meters
        self.pc_range = np.array(pc_range, dtype=np.float64)
        self.img_shape = (640, 480)

        # self.scan_names = glob.glob(osp.join(self.data_root, '*.jpg'))
        self.scan_names = []
        subscenes_list = f'{self.data_root}/{self.split}_files_split_9_1.txt'
        print(f'subscenes_list: {subscenes_list}')
        with open(subscenes_list, 'r') as f:
            self.used_subscenes = f.readlines()
            for i in range(len(self.used_subscenes)):
                name = self.used_subscenes[i].strip()
                self.scan_names.append(f'{self.data_root}/' + self.used_subscenes[i].strip())
        # self.scan_names = glob.glob(osp.join(self.label_root, '*.pkl'))
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        print(f'{split}_subscenes_list_path: {subscenes_list}')
        print(f'{split}集数据: {len(self.scan_names)}')

        # self.depth_eval_transform = T.Compose([Resize(
        #     width=518,
        #     height=518,
        #     resize_target=False,
        #     keep_aspect_ratio=True,
        #     ensure_multiple_of=14,
        #     resize_method='lower_bound',
        #     image_interpolation_method=cv2.INTER_CUBIC,
        # ),
        #     NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     PrepareForNet(),
        # ])

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        filename = osp.basename(self.scan_names[idx])[:-4]
        scene_name = self.scan_names[idx].split('/')[-3]
        # filename = 'NYU0001_0000'
        # print(f'filename: {filename}')
        # print(f'scene_name: {scene_name}')

        filepath = osp.join(self.data_root, scene_name, 'voxels', filename + '.pkl')
        # print(f'filepath: {filepath}')
        with open(filepath, 'rb') as f:
            data_occ = pickle.load(f)

        # print(data_occ.keys())

        # for key, value in data_occ.items():
        #     if key == 'cam_pose':
        #         print(f'key: {key}, value: {value}')
        #     elif key == 'intrinsic':
        #         print(f'key: {key}, value: {value}')
        #     elif key == 'voxel_origin':
        #         print(f'key: {key}, value: {value}')
        #     elif isinstance(value, str):
        #         print(f'key: {key}, value: {value}')
        #     else:
        #         print(f'key: {key}, value.shape: {value.shape}')

        label = {}
        data = {}
        data['filename'] = filename
        data['scene'] = scene_name
        cam_pose = np.linalg.inv(data_occ['cam_pose'])
        data['cam_pose'] = cam_pose
        vox_origin = list(data_occ["voxel_origin"])
        voxel_origin = np.array(vox_origin)
        data['voxel_origin'] = voxel_origin
        cam_K = data_occ['intrinsic']

        # Following SSC literature, the output resolution on NYUv2 is set to 1/4
        if self.use_crop and self.voxel_size == 0.08:
            # [50, 50, 25] 裁切
            target_1_4 = data_occ['target_1_4']

            # 把 >=12 并且 <255 的值都变成 0
            # target_1_4[(target_1_4 >= 12) & (target_1_4 < 255)] = 0

            # 获取target_1_4的唯一值
            # unique_values = np.unique(target_1_4)
            # print(f'target.unique_values: {unique_values}')

            # print(f"target_1_4 中的唯一值: {unique_values}")

            target = target_1_4[:50, :50, :25]
            # target = target_1_4
            target = np.swapaxes(target, 0, 1)
            target[target == 6] = 0
            target[target == 7] = 0
            target[(target >= 8) & (target < 255)] -= 2
            # unique_values2 = np.unique(target)
            # print(f'target.after.unique_values: {unique_values2}')
        else:
            raise ValueError(f'voxel_size: {self.voxel_size} not supported')

        label['target'] = target

        # CP_mega_matrix = compute_CP_mega_matrix(target_1_4, is_binary=False)
        # label['CP_mega_matrix'] = CP_mega_matrix

        # compute the 3D-2D mapping
        projected_pix, fov_mask, pix_z = vox2pix(cam_pose, cam_K, voxel_origin,
                                                 self.voxel_size, self.img_shape, self.scene_size,
                                                 self.pc_range, filepath)

        # print(f'projected_pix.shape: {projected_pix.shape}')
        # print(f'fov_mask.shape: {fov_mask.shape}')
        # print(f'pix_z.shape: {pix_z.shape}')

        data['projected_pix_1'] = projected_pix
        data['fov_mask_1'] = fov_mask
        data['label_mask'] = target != 255

        # compute the masks, each indicates voxels inside a frustum
        frustums_masks, frustums_class_dists = compute_local_frustums(
            projected_pix,
            pix_z,
            target,
            self.img_shape,
            n_classes=self.num_classes,
            size=self.frustum_size,
        )
        label['frustums_masks'] = frustums_masks
        label['frustums_class_dists'] = frustums_class_dists

        img_path = osp.join(self.data_root, scene_name, 'color', filename + '.jpg')
        img = Image.open(img_path).convert('RGB')
        img_W, img_H = img.size
        img = img.resize(((640, 480)))

        W_factor = self.img_shape[0] / img_W
        H_factor = self.img_shape[1] / img_H
        scaled_cam_K = cam_K.copy()
        scaled_cam_K[0, 0] *= W_factor  # fx
        scaled_cam_K[0, 2] *= W_factor  # cx
        scaled_cam_K[1, 1] *= H_factor  # fy
        scaled_cam_K[1, 2] *= H_factor  # cy

        data['cam_K'] = scaled_cam_K[:3, :3]

        img = np.asarray(img, dtype=np.float32) / 255.0

        data['img'] = self.transforms(img)  # (3, H, W)
        # data['img'] = self.depth_eval_transform({'image': img})['image']  # (3, H, W)

        data['use_depth_eval'] = self.use_depth_eval
        if not self.use_depth_eval:
            depth_path = osp.join(self.data_root, scene_name, 'depth', filename + '.png')
            depth = Image.open(depth_path)
            depth = depth.resize(((640, 480)))
            depth_np = np.array(depth) / 1000.  # noqa
            data['depth'] = depth_np

        color_im = img
        if self.use_tsdf:
            vol_bnds = np.zeros((3, 2))
            vol_bnds[:, 0] = voxel_origin
            vol_bnds[:, 1] = voxel_origin + np.array(self.scene_size)
            tsdf_volume = TSDFVolume(vol_bnds=vol_bnds, voxel_size=self.voxel_size, use_gpu=False)
            depth_im = data['depth'][0] if len(data['depth'].shape) == 3 else data['depth']
            tsdf_volume.integrate(color_im=color_im, depth_im=depth_im, cam_intr=self.cam_K,
                                  cam_pose=np.linalg.inv(cam_pose))
            vox_tsdf, vox_tsdf_color = tsdf_volume.get_volume()
            data['vox_tsdf'] = vox_tsdf[np.newaxis, ...]

        def ndarray_to_tensor(data: dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    if v.dtype == np.float64:
                        v = v.astype('float32')
                    data[k] = torch.from_numpy(v)

        ndarray_to_tensor(data)
        ndarray_to_tensor(label)
        return data, label
