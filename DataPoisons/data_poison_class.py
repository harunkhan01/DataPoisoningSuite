class DataPoisonClass:
    def __init__(self, cfg):
        self.attack_name = cfg.name
        self.poison_ratio = cfg.ratio