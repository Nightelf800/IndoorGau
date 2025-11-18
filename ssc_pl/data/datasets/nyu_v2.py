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
class NYUv2(Dataset):

    META_INFO = {
        'class_weights':
        torch.tensor((0.05, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
        'class_names': ('empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa',
                        'table', 'tvs', 'furn', 'objs'),
    }

    def __init__(self, split, data_root, label_root, voxel_size=0.08, pc_range=None, depth_root=None,
                 use_crop=True, frustum_size=4, depth_eval=False, depth_encoder='null', use_tsdf=False):
        self.data_root = osp.join(data_root, 'NYU' + split)
        self.label_root = osp.join(label_root, 'NYU' + split)
        self.depth_root = osp.join(depth_root, 'NYU' + split) if depth_root else None
        self.depth_eval = depth_eval
        self.frustum_size = frustum_size
        self.num_classes = 12
        self.use_tsdf = use_tsdf
        # self.ckpt_path = '/share/lkl/Symphonies/outputs/11_19_dim64_sym/e25_miou0.2860.ckpt'
        # self.meta_info = {}
        # self.meta_info['class_weights'] = torch.tensor([0.0500, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        # 1.0000, 1.0000, 1.0000])
        # self.meta_info['class_names'] = ('empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa', 'table', 'tvs', 'furn', 'objs')
        # self.cfg = ConfigManager.get_global_cfg()
        # self.symphony_model = LitModule.load_from_checkpoint(self.ckpt_path, **self.cfg, meta_info=self.meta_info)
        # device = torch.device('cuda')
        # self.upsampler = JBUFeatUp()
        # checkpoint = torch.load('/share/lkl/Symphonies/checkpoints/maskclip_jbu_stack_cocostuff.ckpt')
        # self.upsampler.load_state_dict(checkpoint['state_dict']).to(device)
        # self.upsampler.eval()
        self.voxel_size = voxel_size  # meters
        self.use_crop = use_crop    # crop or scale

        self.scene_size = (4.8, 4.8, 2.88)  # meters
        # self.scene_size = (4, 4, 2)  # meters
        self.pc_range = np.array(pc_range, dtype=np.float64)
        self.img_shape =  (640, 480)
        self.cam_K = np.array(((518.8579, 0, 320), (0, 518.8579, 240), (0, 0, 1)))

        self.scan_names = glob.glob(osp.join(self.data_root, '*.bin'))
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        xyz = get_meshgrid([0, 0, 0, 4, 4, 2], self.voxel_size)
        self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1) # x, y, z, 4

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
        # filename = 'NYU0001_0000'

        filepath = osp.join(self.label_root, filename + '.pkl')
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        label = {}
        data['filename'] = filename
        cam_pose = np.linalg.inv(data['cam_pose'])
        data['cam_pose'] = cam_pose
        voxel_origin = data['voxel_origin']
        data['cam_K'] = self.cam_K
        data['voxel_size'] = np.float32(self.voxel_size)
        data['image_wh'] = np.array(self.img_shape)[np.newaxis, :]
        
        # camera extrinsic matual intrinsic
        M_intrinsic = np.eye(4)
        M_intrinsic[:3, :3] = self.cam_K
        data['projection_mat'] = np.matmul(M_intrinsic, cam_pose)

        # Following SSC literature, the output resolution on NYUv2 is set to 1/4
        if self.use_crop and self.voxel_size == 0.08:
            # [50, 50, 25] 裁切
            target_1_4 = data.pop('target_1_4').transpose(0, 2, 1)
            target = target_1_4[:50, :50, :25]
        elif self.use_crop and self.voxel_size == 0.04:
            # [100, 100, 50] 裁切
            target_1_2 = data.pop('target_1_2').transpose(0, 2, 1)
            target = target_1_2[:100, :100, :50]
        elif not self.use_crop and self.voxel_size == 0.08:
            # [50, 50, 25] 缩放
            target = data.pop('target_1_4').transpose(0, 2, 1)
        else:
            # [100, 100, 50] 缩放
            target = data.pop('target_1_1').transpose(0, 2, 1)

        data['xyz'] = self.xyz[..., :3] + voxel_origin + self.voxel_size * 0.5
        label['target'] = target
        target_1_4 = data.pop('target_1_16').transpose(0, 2, 1)

        CP_mega_matrix = compute_CP_mega_matrix(target_1_4, is_binary=False)
        label['CP_mega_matrix'] = CP_mega_matrix



        # compute the 3D-2D mapping
        projected_pix, fov_mask, pix_z = vox2pix(cam_pose, self.cam_K, voxel_origin,
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

        img_path = osp.join(self.data_root, filename + '_color.jpg')
        img = Image.open(img_path).convert('RGB')
        img = img.resize(((640, 480)))
        # device = torch.device('cuda')
        # transform = T.Compose([T.Resize((432, 768)), T.ToTensor(), norm])
        # img_tensor = transform(img).unsqueeze(0).to(device)
        # with torch.no_grad():
        #         hr_feats = upsampler(image_tensor)
        # import pdb;
        # pdb.set_trace()
        img = np.asarray(img, dtype=np.float32) / 255.0

        data['img'] = self.transforms(img)  # (3, H, W)
        # data['img'] = self.depth_eval_transform({'image': img})['image']  # (3, H, W)


        if self.depth_root is not None:
            if self.depth_eval is False:
                data['depth_eval'] = False
                depth_path = osp.join(self.depth_root, filename + '.png')
                depth = Image.open(depth_path)
                depth = depth.resize(((640, 480)))
                data['depth'] = np.array(depth) / 8000.  # noqa
            else:
                data['depth_eval'] = True
                # 本地加载深度数据集，若使用模型直接推理，请注释这段代码
                # depth_path = osp.join(self.depth_root, filename + '_pred.png')
                # depth = Image.open(depth_path)
                # data['depth'] = np.array(depth) / 8000.
        color_im = img
        if self.use_tsdf:
            vol_bnds = np.zeros((3, 2))
            vol_bnds[:, 0] = voxel_origin
            vol_bnds[:, 1] = voxel_origin + np.array(self.scene_size)
            tsdf_volume = TSDFVolume(vol_bnds=vol_bnds, voxel_size=self.voxel_size, use_gpu=False)
            depth_im = data['depth'][0] if len(data['depth'].shape) == 3 else data['depth']
            tsdf_volume.integrate(color_im=color_im, depth_im=depth_im, cam_intr=self.cam_K, cam_pose=np.linalg.inv(cam_pose))
            vox_tsdf, vox_tsdf_color = tsdf_volume.get_volume()
            data['vox_tsdf'] = vox_tsdf[np.newaxis, ...]
            
        def ndarray_to_tensor(data: dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    if v.dtype == np.float64:
                        v = v.astype('float32')
                    data[k] = torch.from_numpy(v)


            # ('depth', 'cam_K', 'cam_pose', 'voxel_origin', f'projected_pix_{self.volume_scale}',
            #      f'fov_mask_{self.volume_scale}')))
            # img
        

        # data['voxel_feat'] = None
        # print(type(data['depth_eval']))
        # print(data['depth_eval'])

        # outs = self.symphony_model(data)
        # data['voxel_feat'] = outs['voxel_feat']
        # import pdb;
        # pdb.set_trace()
        ndarray_to_tensor(data)
        ndarray_to_tensor(label)
        return data, label
