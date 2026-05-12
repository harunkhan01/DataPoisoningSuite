from .finetuning.factory import fine_tuning_factory
from .data_poisons.factory import data_poison_factory
from .evaluator.evaluate_class import EvaluatePoisons
from .data_loading.factory import data_loading_factory
from .data_loading.poison_dataset import PoisonedDataset
from .model_loading.factory import model_loading_factory


class ExperimentRunner:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self):
        self.run()

    def run(self):
        device = self.cfg.device

        # stage 1 -- construct dataset
        dataset = data_loading_factory(self.cfg.dataset)

        print(f'Constructed dataset...')

        # stage 2 -- construct model and corresponding transforms
        model, model_transforms = model_loading_factory(self.cfg.model)

        print(f'Constructed model and model transform...')

        # stage 3 -- select and build poison 
        poison_obj = data_poison_factory(self.cfg.poison)

        print(f'Constructed poison...')

        # stage 4 -- construct poisoned dataset
        poison_train = PoisonedDataset(dataset.train, poison_obj, self.cfg.poison)
        poison_test  = PoisonedDataset(dataset.test, poison_obj, self.cfg.poison)

        # stage 5 -- finetune model with poisoned dataset
        fine_tuning_factory(model, poison_train, device, self.cfg.training)

        print(f'Finished finetuning...')

        # stage 6 -- evaluate poison performance 
        eval_obj = EvaluatePoisons(model, dataset, poison_test, device)
        benign_acc, poison_acc = eval_obj.evaluate()

        print(f'Evaluation complete.')

        # stage 7 -- save experiment results (will implement later)

        print(f'All done.')       

