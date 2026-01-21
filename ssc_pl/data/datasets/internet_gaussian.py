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

from ...utils.helper import vox2pix, compute_local_frustums, compute_CP_mega_matrix, get_meshgrid
from depth_eval.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet




class InterNetGaussian(Dataset):

    META_INFO = {
        'class_weights':
        torch.tensor((0.05, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
        'class_names': ('empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa',
                        'table', 'tvs', 'furn', 'objs'),
    }

    def __init__(self, split, data_root, voxel_size=0.08, pc_range=None, 
        target_size=1036, frustum_size=4):
        self.data_root = data_root

        self.frustum_size = frustum_size
        self.num_classes = 12

        self.voxel_size = voxel_size  # meters
        self.target_size = target_size

        # self.scene_size = (4.8, 4.8, 2.88)  # meters
        # self.scene_size = (4, 4, 2)  # meters
        self.pc_range = np.array(pc_range, dtype=np.float64)

        with open(osp.join(self.data_root, 'image_paths.txt'), 'r') as f:
            self.img_paths = f.readlines()
        self.img_paths = [p.strip() for p in self.img_paths]
        
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # xyz = get_meshgrid([0, 0, 0, 4, 4, 2], self.voxel_size)
        # self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1) # x, y, z, 4


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # filename = osp.basename(self.img_paths[idx])[:-4]
        # filename = 'NYU0001_0000'
        data = {}
        label = {}

        voxel_origin = np.array([0.0, 0.0, 0.0])
        data['voxel_origin'] = voxel_origin

        # data['image_wh'] = np.array((640, 480))[np.newaxis, :]
        

        # data['xyz'] = self.xyz[..., :3] + voxel_origin + self.voxel_size * 0.5

        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB') 
        original_width, original_height = img.size

        # if original_width >= self.target_size and original_height >= self.target_size:
        #     crop_size = self.target_size
        #     left = (original_width - crop_size) // 2
        #     top = (original_height - crop_size) // 2
        #     right = left + crop_size
        #     bottom = top + crop_size    
        #     img = img.crop((left, top, right, bottom))
        new_width = self.target_size
        new_height = round(original_height * (new_width / original_width) / 14) * 14
        img = img.resize((new_width, new_height), Image.BILINEAR)
        # if original_height >= self.target_size:
        #     start_y = (new_height - self.target_size) // 2
        #     img = img.crop((0, start_y, new_width, start_y + self.target_size))
        # else:
        #     raise ValueError(f'Image {img_path} is smaller than target size {self.target_size}')
        # print(f'img.size: {img.size}')
        
        updated_width, updated_height = img.size
        data['image_wh'] = np.array((updated_width, updated_height))[np.newaxis, :]

        img = np.asarray(img, dtype=np.float32) / 255.0
        data['img'] = self.transforms(img)  # (3, H, W)
        # print(f'dataloader data[img].shape: {data["img"].shape}')
        label['img'] = img.transpose(2, 0, 1)  # (3, H, W)
        # data['img'] = self.depth_eval_transform({'image': img})['image']  # (3, H, W)

        

        def ndarray_to_tensor(data: dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    if v.dtype == np.float64:
                        v = v.astype('float32')
                    data[k] = torch.from_numpy(v)

        ndarray_to_tensor(data)
        ndarray_to_tensor(label)

        # print(f'label[img].shape: {label["img"].shape}')
        # print(f'label[depth].shape: {label["depth"].shape}')

        return data, label
