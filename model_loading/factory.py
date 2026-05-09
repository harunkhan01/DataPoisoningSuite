from .registry import MODEL_REGISTRY
import specific_models


def model_loading_factory(cfg):

    model_name = cfg.name

    model_class = MODEL_REGISTRY[model_name]

    model = model_class()

    return model