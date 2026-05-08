"""
Implements a wrapper around torch models to provide a way to load models and perform
operations on models (detaching classifier heads and so on)

"""
import torch
import torchvision.models as models

class ModelLoaderClass:
    def __init__(self, model_type, output_classes, weights="DEFAULT"):
        self.model = None
        self.weights = None
        self.construct_model(model_type, output_classes, weights)
        self.preprocess = self.weights.transforms()

    def load_model(self, model_type, output_classes, weights):
        if "resnet18" in model_type:
            self.weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)

            if output_classes is not None:
                self.model.fc = torch.nn.Linear(self.model.fc.in_features, output_classes)

        elif "resnet50" in model_type:
            self.weights = models.ResNet50_Weights.DEFAULT
            self.model = models.resnet50(weights=weights)

            if output_classes is not None:
                self.model.fc = torch.nn.Linear(self.model.fc.in_features, output_classes)

        elif "vgg16" in model_type:
            self.weights = models.VGG16_Weights.DEFAULT
            self.model = models.vgg16(weights=weights)

            if output_classes is not None:
                self.model.classifier[6] = torch.nn.Linear(
                    self.model.classifier[6].in_features,
                    output_classes
                )

        elif "mobilenet_v3_small" in model_type:
            # Small + efficient network
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            model = models.mobilenet_v3_small(weights=weights)

            if output_classes is not None:
                model.classifier[3] = torch.nn.Linear(
                    model.classifier[3].in_features,
                    output_classes
                )
                
        else:
            raise ValueError(f"Error! Model Type {model_type} is not supported.")        