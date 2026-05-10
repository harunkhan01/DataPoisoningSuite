from .DataLoading.factory import data_loading_factory
from .ModelLoading.factory import model_loading_factory
from .DataPoisons.factory import data_poison_factory
from .FineTuning.factory import fine_tuning_factory

class ExperimentRunner:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self):
        self.run()

    def run(self):
        dataset = data_loading_factory(self.cfg.dataset)

        print(f'Constructed dataset...')

        model, model_transforms = model_loading_factory(self.cfg.model)

        print(f'Constructed model and model transform...')

        poison = data_poison_factory(self.cfg.poison)

        print(f'Constructed poison...')

        finetuner = fine_tuning_factory(self.cfg.training)

        print(f'Finished finetuning...')

        