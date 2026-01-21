# -*- coding: utf-8 -*-
import os
os.environ['NCCL_P2P_DISABLE'] = '1'
import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks
from cfg_module import ConfigManager
import torch
import random
import numpy as np

def set_seed(seed=0):
    """固定随机种子以确保实验的可重复性"""
    random.seed(seed)  # Python内置的random模块
    np.random.seed(seed)  # numpy库
    torch.manual_seed(seed)  # CPU上的PyTorch操作
    torch.cuda.manual_seed(seed)  # 当前GPU上的PyTorch操作
    torch.cuda.manual_seed_all(seed)  # 所有GPU上的PyTorch操作
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 如果网络输入数据维度或类型上变化不大，设置为True可以增加运行效率
    os.environ['PYTHONHASHSEED'] = str(seed)  # 通过环境变量固定Python哈希算法的种子

@hydra.main(config_path='configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    set_seed(0)

    if os.environ.get('LOCAL_RANK', 0) == 0:
        print(OmegaConf.to_yaml(cfg))
    cfg, callbacks = pre_build_callbacks(cfg)


    ConfigManager.set_global_cfg(cfg)
    # sym_model = LitModule.load_from_checkpoint(ckpt_path, **cfg, meta_info=meta_info)
    # ConfigManager.set_global_model(sym_model)
    # import pdb
    # pdb.set_trace()
    dls, meta_info = build_data_loaders(cfg.data)

    # model = LitModule(**cfg, **meta_info)


    if cfg.get('ckpt_path'):
        print(f"Loading weights from: {cfg.ckpt_path}")
        model = LitModule.load_from_checkpoint(cfg.ckpt_path, **cfg, **meta_info)
    else:
        print("No pre-trained weights")
        model = LitModule(**cfg, **meta_info)

    trainer = L.Trainer(strategy='ddp', **cfg.trainer, **callbacks)
    trainer.fit(model, *dls[:2])


if __name__ == '__main__':
    main()


