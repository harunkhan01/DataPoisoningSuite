from torchvision.transforms import Compose

from .registry import TRANSFORM_REGISTRY
from . import transforms

class GenericDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.transform = self.build_transform()
        self.build()
    
    def build_transform(self):
        transform_dict = self.cfg['transforms']

        to_tensor = TRANSFORM_REGISTRY['ToTensor']
        op_list = [to_tensor()]

        for name, args in transform_dict.items():
            fn = TRANSFORM_REGISTRY[name]
            op_list.append(fn(**args))

        return Compose(op_list)