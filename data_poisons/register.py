POISON_REGISTRY = {}

def register_poison(name):

    def wrapper(cls):
        POISON_REGISTRY[name] = cls
        return cls

    return wrapper