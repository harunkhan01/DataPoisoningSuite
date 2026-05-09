import torch
import torchvision.models as models

from ..base import GenericModel
from ..registry import register_model

@register_model('vgg16')
class VGG16(GenericModel):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build(self):
        self.model = self.build_vgg16()

    def build_vgg16(self):
        self.weights = models.VGG16_Weights.DEFAULT
        self.model = models.vgg16(weights=self.weights)
        self.model.classifier[6] = torch.nn.Linear(
                    self.model.classifier[6].in_features,
                    self.output_classes
                )