from mmseg.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class BaseBackbone(BaseModule):

    """Base backbone class.
    image backbone -> neck -> lifter -> encoder -> segmentor
    """

    def __init__(self, init_cfg=None, **kwargs) -> None:
        super().__init__(init_cfg)
    
    def forward(
        self, 
        **kwargs
    ):
        pass