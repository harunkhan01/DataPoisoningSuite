import os

from .finetuning.factory import fine_tuning_factory
from .data_poisons.factory import data_poison_factory
from .evaluator.evaluator_class import EvaluatePoisons
from .data_loading.factory import data_loading_factory
from .data_loading.poison_dataset import PoisonedDataset
from .model_loading.factory import model_loading_factory


class ExperimentRunner:
    def __init__(self, cfg, file_name):
        self.cfg = cfg
        self.file_name = file_name # used to save the run

    def __call__(self):
        self.run()

    def run(self):
        device = self.cfg['device']

        # stage 1 -- construct dataset
        dataset = data_loading_factory(self.cfg['dataset'])

        print(f'Constructed dataset...')

        # stage 2 -- construct model and corresponding transforms
        model, model_transforms = model_loading_factory(self.cfg['model'])

        print(f'Constructed model and model transform...')

        # stage 3 -- select and build poison 
        poison_obj = data_poison_factory(self.cfg['poison'])

        print(f'Constructed poison...')

        # stage 4 -- construct poisoned dataset
        poison_train = PoisonedDataset(dataset.train, poison_obj, self.cfg['poison'])
        poison_test  = PoisonedDataset(dataset.test, poison_obj, self.cfg['poison'], True)

        # stage 5 -- finetune model with poisoned dataset
        fine_tuning_factory(model, poison_train, poison_test, device, self.cfg['finetuning'])

        print(f'Finished finetuning...')

        # stage 6 -- evaluate poison performance 
        eval_obj = EvaluatePoisons(model, dataset.test, poison_test, device)
        benign_acc, poison_acc = eval_obj.evaluate()

        print(f'Evaluation complete.')

        # stage 7 -- save experiment results (will implement later)
        save_path = os.path.join('./outputs', self.file_name)

        with open(save_path, 'a') as f:
            f.write(f'Benign Accuracy: {benign_acc}, Poison Accuracy: {poison_acc}')

        print(f'All done. Closing...')