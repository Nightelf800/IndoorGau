import os.path as osp
import glob
import re
import numpy as np
import torch
import pickle
import copy
import cv2
from PIL import Image
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T

from ...utils.helper import vox2pix, compute_local_frustums
from ...utils.fusion import TSDFVolume

# iso original
# empty_target_list = [
#     "gathered_data/scene0067_01/00051.pkl",
#     "gathered_data/scene0083_00/00095.pkl",
#     "gathered_data/scene0128_00/00030.pkl",
#     "gathered_data/scene0301_02/00098.pkl",
#     "gathered_data/scene0301_02/00099.pkl",
#     "gathered_data/scene0375_00/00046.pkl",
#     "gathered_data/scene0375_00/00047.pkl",
#     "gathered_data/scene0378_01/00027.pkl",
#     "gathered_data/scene0448_01/00059.pkl",
#     "gathered_data/scene0448_01/00060.pkl",
#     "gathered_data/scene0448_01/00064.pkl",
#     "gathered_data/scene0448_01/00066.pkl",
#     "gathered_data/scene0505_01/00082.pkl",
#     "gathered_data/scene0538_00/00043.pkl",
#     "gathered_data/scene0538_00/00044.pkl",
#     "gathered_data/scene0625_01/00001.pkl",
#     "gathered_data/scene0642_01/00090.pkl",
#     "gathered_data/scene0674_00/00052.pkl",
#     "gathered_data/scene0684_00/00043.pkl",
#     "gathered_data/scene0702_01/00027.pkl",
#     "gathered_data/scene0702_01/00028.pkl",
#     "gathered_data/scene0702_01/00029.pkl",
#     "gathered_data/scene0702_01/00030.pkl",
#     "gathered_data/scene0702_01/00031.pkl",
#     # val set
#     "gathered_data/scene0067_01/00052.pkl",
#     "gathered_data/scene0121_01/00004.pkl",
#     "gathered_data/scene0121_01/00005.pkl",
#     "gathered_data/scene0128_00/00029.pkl",
#     "gathered_data/scene0286_01/00087.pkl",
#     "gathered_data/scene0301_02/00097.pkl",
#     "gathered_data/scene0375_00/00048.pkl",
#     "gathered_data/scene0448_01/00058.pkl",
#     "gathered_data/scene0448_01/00061.pkl",
#     "gathered_data/scene0448_01/00062.pkl",
#     "gathered_data/scene0448_01/00063.pkl",
#     "gathered_data/scene0448_01/00065.pkl",
#     "gathered_data/scene0538_00/00083.pkl",
#     "gathered_data/scene0674_00/00053.pkl"
# ]

# by scenes
empty_target_list = [
    "gathered_data/scene0067_01/00050.pkl",
    "gathered_data/scene0128_00/00029.pkl",
    "gathered_data/scene0301_02/00093.pkl",
    "gathered_data/scene0301_02/00096.pkl",
    "gathered_data/scene0301_02/00098.pkl",
    "gathered_data/scene0301_02/00099.pkl",
    "gathered_data/scene0375_00/00062.pkl",
    "gathered_data/scene0448_01/00064.pkl",
    "gathered_data/scene0448_01/00065.pkl",
    "gathered_data/scene0449_02/00054.pkl",
    "gathered_data/scene0449_02/00055.pkl",
    "gathered_data/scene0625_01/00012.pkl",
    "gathered_data/scene0642_01/00083.pkl",
    "gathered_data/scene0674_00/00046.pkl",
    "gathered_data/scene0702_01/00032.pkl",
    "gathered_data/scene0702_01/00033.pkl",
    # val set
    "gathered_data/scene0067_01/00049.pkl",
    "gathered_data/scene0128_00/00030.pkl",
    "gathered_data/scene0286_01/00084.pkl",
    "gathered_data/scene0301_02/00094.pkl",
    "gathered_data/scene0301_02/00095.pkl",
    "gathered_data/scene0301_02/00097.pkl",
    "gathered_data/scene0375_00/00061.pkl",
    "gathered_data/scene0448_01/00063.pkl",
    "gathered_data/scene0538_00/00060.pkl",
    "gathered_data/scene0642_01/00084.pkl",
    "gathered_data/scene0683_00/00093.pkl",
    "gathered_data/scene0684_00/00050.pkl",
]


def img_transform(crop, flip):
    ida_rot = torch.eye(2)
    ida_tran = torch.zeros(2)

    ida_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b

    ida_mat = ida_rot.new_zeros(4, 4)
    ida_mat[3, 3] = 1
    ida_mat[2, 2] = 1
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 3] = ida_tran

    return ida_mat


class ScanNet(Dataset):
    META_INFO = {
        'class_weights':
            torch.tensor((0.05, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
        'class_names': ('empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa',
                        'table', 'tvs', 'furn', 'objs'),
    }

    def __init__(self, split, data_root, voxel_size=0.08, depth_root=None,
                 use_crop=False, frustum_size=4, depth_eval=False, use_tsdf=False, img_shape=[640, 480], use_hvm=False,
                 use_occdepth=False, pc_range=None):
        if split == 'test':
            split = 'val'
        self.split = split
        self.data_root = data_root
        self.img_shape = img_shape
        self.voxel_size = voxel_size
        self.frustum_size = frustum_size
        self.use_tsdf = use_tsdf
        self.use_hvm = use_hvm
        self.use_crop = use_crop
        self.use_occdepth = use_occdepth
        assert self.voxel_size == 0.08, Exception('only support 0.08m voxel size for Occ-scannet dataset')
        assert self.use_crop is False, Exception("Crop is not supported for Occ-scannet dataset")
        self.scene_size = (4.8, 4.8, 2.88) if not use_crop else (4, 4, 2)
        self.depth_eval = depth_eval
        self.pc_range = pc_range if pc_range is not None else [0, 0, 0, 0, 0, 0]
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.data_list = []
        subscenes_list = f'{self.data_root}/{self.split}_subscenes.txt'
        print(f'subscenes_list: {subscenes_list}')
        with open(subscenes_list, 'r') as f:
            self.used_subscenes = f.readlines()
            for i in range(len(self.used_subscenes)):
                name = self.used_subscenes[i].strip()
                if name in empty_target_list:
                    continue
                self.data_list.append(f'{self.data_root}/' + self.used_subscenes[i].strip())
        print(split, len(self.data_list))

        # 用来查找一些脏数据
        # debug
        # from tqdm import tqdm
        # for i in tqdm(range(len(self.used_subscenes))):
        #     name = self.used_subscenes[i]
        #     with open(name, 'rb') as f:
        #         data_item = pickle.load(f)
        #     target = data_item['target_1_4']
        #     nonempty_target = target[np.logical_and(target != 255, target != 0)].sum()
        #     if nonempty_target == 0:
        #         print(name)
        # print("pass")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        name = self.data_list[idx]

        # 使用正则表达式提取场景名称
        match = re.search(r'scene\d+_\d+', name)
        scene_name = match.group(0) if match else None

        # 找到最后一个 '/' 的位置
        last_slash_index = name.rfind('/')
        # 找到 '.pkl' 的位置
        pkl_index = name.find('.pkl', last_slash_index)
        # 提取两个位置之间的内容
        sample_name = name[last_slash_index + 1: pkl_index]

        # print(f'name: {name}, scene_name: {scene_name}')

        with open(name, 'rb') as f:
            data_item = pickle.load(f)

        cam_pose = data_item["cam_pose"]
        world_2_cam = np.linalg.inv(cam_pose)
        cam_intrin = data_item['intrinsic'].copy()

        img = cv2.imread(
            data_item['img'].replace("/home/hongxiao.yu/projects/mmdetection3d/data/scannet", self.data_root))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_H, img_W = img.shape[0], img.shape[1]
        img = cv2.resize(img, (640, 480))
        W_factor = 640.0 / img_W
        H_factor = 480.0 / img_H
        cam_intrin[0] *= W_factor
        cam_intrin[1] *= H_factor
        img_H, img_W = img.shape[0], img.shape[1]

        if img.shape[0] != self.img_shape[1]:
            img = cv2.resize(img, self.img_shape)
            W_factor = self.img_shape[0] / img_W
            H_factor = self.img_shape[1] / img_H
            cam_intrin[0] *= W_factor
            cam_intrin[1] *= H_factor
            img_H, img_W = img.shape[0], img.shape[1]

        img_for_network = np.asarray(img, dtype=np.float32) / 255.00
        img_for_network = self.transforms(img_for_network)

        cam_K = cam_intrin[:3, :3][None]

        vox_origin = list(data_item["voxel_origin"])
        vox_origin = np.array(vox_origin)

        if self.depth_eval:
            depth_img = None
        else:
            depth_img = (Image.open(data_item['depth_gt']
                                    .replace("/home/hongxiao.yu/projects/mmdetection3d/data/scannet", self.data_root))
                         .convert('I;16'))
            depth_img = np.array(depth_img) / 1000.0

        if depth_img.shape[0] != self.img_shape[1]:
            depth_img = cv2.resize(depth_img, self.img_shape)

        projected_pix, fov_mask, pix_z = vox2pix(
            world_2_cam,
            cam_intrin,
            vox_origin,
            self.voxel_size,
            (img_W, img_H),
            self.scene_size,
            self.pc_range,
            name
        )

        if self.voxel_size == 0.08:
            target = data_item[
                "target_1_4"
            ]  # 60,60,36
            if self.use_crop:
                assert Exception("Crop is not implemented  for Occ-scannet")
        # 获取target_1_4的唯一值
        # unique_values = np.unique(target)
        # print(f"target_1_4 中的唯一值: {unique_values}")

        target = np.where(target == 255, 0, target)
        target = np.swapaxes(target, 0, 1)
        # print(f"target.shape: {target.shape}")

        frustums_masks, frustums_class_dists = compute_local_frustums(
            projected_pix,
            pix_z,
            target,
            (img_W, img_H),
            n_classes=12,
            size=self.frustum_size,
        )

        data = {
            'img': img_for_network,
            # 'raw_img': raw_img,
            'depth': depth_img,
            'projected_pix_1': projected_pix,
            'fov_mask_1': fov_mask,
            'cam_pose': world_2_cam,
            'cam_K': cam_K[0],
            'voxel_origin': vox_origin,
            # 用于可视化
            'name': name,
            'scene': scene_name,
            'sample_name': sample_name,
            'cam_intrinsic': data_item['intrinsic'],
        }

        if self.use_tsdf:
            vol_bnds = np.zeros((3, 2))
            vol_bnds[:, 0] = vox_origin
            vol_bnds[:, 1] = vox_origin + np.array(self.scene_size)
            tsdf_volume = TSDFVolume(vol_bnds=vol_bnds, voxel_size=self.voxel_size, use_gpu=False)
            tsdf_volume.integrate(color_im=img, depth_im=data['depth'], cam_intr=data['cam_K'], cam_pose=cam_pose)
            vox_tsdf, vox_tsdf_color = tsdf_volume.get_volume()
            data['vox_tsdf'] = vox_tsdf[np.newaxis, ...]

        label = {
            'target': target,
            'frustums_masks': frustums_masks,
            'frustums_class_dists': frustums_class_dists,
        }

        if self.use_hvm:
            if self.voxel_size == 0.08 and self.use_crop:
                label['lga'] = data_item.pop('lga_1_4_crop')
            elif self.voxel_size == 0.08 and not self.use_crop:
                label['lga'] = data_item.pop('lga_1_4')
            else:
                raise Exception(f"voxel_size {self.voxel_size} not supported")

        def ndarray_to_tensor(data: dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    if v.dtype == np.float64:
                        v = v.astype('float32')
                    data[k] = torch.from_numpy(v)

        ndarray_to_tensor(data)
        ndarray_to_tensor(label)

        return data, label