from torchvision.transforms import Compose

from .registry import TRANSFORM_REGISTRY
import transforms

class GenericDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.transform = self.build_transform()
        self.build()
    
    def build_transform(self):
        transform_dict = self.cfg.transforms

        op_list = []

        for name, args in transform_dict.items():
            fn = TRANSFORM_REGISTRY[name]
            op_list.append(fn(**args))

        return Compose(op_list)