
from .DataLoading.data_loader_class import DataLoaderClass
from .ModelLoading.model_loader_class import ModelLoaderClass

def main():

    # stages of execution -- we can set up an argparse to have more programmatic implementation of arguments

    # STAGE 1: Load Model (pretrained)
    model = ModelLoaderClass()

    # STAGE 2: Load Dataset
    dl_obj = DataLoaderClass()

    

    # STAGE 3: Finetune model with poisoned dataset

    # STAGE 4: Evaluate performance

    pass