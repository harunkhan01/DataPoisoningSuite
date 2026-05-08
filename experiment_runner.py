from .DataLoading import data_loader_class
from .ModelLoading import model_loader_class
from .DataPoisons import data_poison_class
from .FineTuning import fine_tuning_class

class ExperimentRunner:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self):
        self.run()

    def run(self):
        dataset = data_loader_class(self.cfg.dataset)

        model = model_loader_class(self.cfg.model)

        poison = data_poison_class(self.cfg.poison)

        finetuner = fine_tuning_class(self.cfg.training)