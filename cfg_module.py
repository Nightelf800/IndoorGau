import torch
# from your_module import LitModule  # 确保正确导入 LitModule

class ConfigManager:
    _global_cfg = None
    _global_model = None  # 添加一个用于存储模型的静态变量

    @staticmethod
    def set_global_cfg(cfg):
        ConfigManager._global_cfg = cfg

    @staticmethod
    def get_global_cfg():
        return ConfigManager._global_cfg

    @staticmethod
    def set_global_model(model):
        ConfigManager._global_model = model

    @staticmethod
    def get_global_model():
        return ConfigManager._global_model