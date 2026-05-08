
from .DataLoading.data_loader_class import DataLoaderClass
from .FineTuning.fine_tuning_class import FineTuningClass
from .ModelLoading.model_loader_class import ModelLoaderClass


def main(**kwargs):

    # stages of execution -- we can set up an argparse to have more programmatic implementation of arguments
    model_type = kwargs.get('model_type', 'mobilenet_v3_small')
    output_classes = kwargs.get('output_classes', 10)
    dataset_name = kwargs.get('dataset', 'mnist')

    # STAGE 1: Load Model (pretrained)
    model_obj = ModelLoaderClass(model_type, output_classes)

    # STAGE 2: Load Dataset
    dl_obj = DataLoaderClass(dataset_name, model_obj.preprocess)

    # STAGE 3: Construct Poison

    # STAGE 4: Inject Poison

    # STAGE 5: Finetune model with poisoned dataset
    ft_obj = FineTuningClass(dl_obj, model_obj)
    ft_obj.fine_tune()

    # STAGE 6: Evaluate performance

    pass