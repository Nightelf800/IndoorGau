import torch
from dinov2.hub.backbones import dinov2_vitb14, dinov2_vitg14, dinov2_vitl14, dinov2_vits14

from PIL import Image
import numpy as np


dinov2_backbone = dinov2_vits14(pretrained=False)
ckpt_path = '/data0/3DXR/liux/02.Code/01.Occ/05.Symphonies/dinov2/dinov2_vits14_pretrain.pth'
# dinov2_backbone = dinov2_vitb14(pretrained=False)
# ckpt_path = "/data0/3DXR/liux/02.Code/01.Occ/05.Symphonies/Symphonies/checkpoints/dinov2_vitb14_pretrain.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dict = torch.load(ckpt_path, map_location=device)
dinov2_backbone.load_state_dict(state_dict, strict=True)
dinov2_backbone.to(device)
dinov2_backbone.eval()

sample_path = '/data0/3DXR/liux/02.Code/01.Occ/05.Symphonies/dinov2/example.jpg'
image = Image.open(sample_path).convert("RGB").resize((630, 476))
image = np.moveaxis(np.array(image), -1, 0) / 255
array = np.expand_dims(image, axis=0).astype('float32')
array = torch.from_numpy(array).cuda()
feature = dinov2_backbone.get_intermediate_layers(
    array, n=[2, 5, 8, 11], return_class_token=True
)

print("====")


# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')