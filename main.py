import yaml
from .experiment_runner import ExperimentRunner

def load_config(config_file):
    with open(config_file, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f'Error parsing YAML file: {e}')

def main(config_file):

    # safe load yaml file

    cfg = load_config(config_file)

    expr = ExperimentRunner(cfg, config_file)
    expr()