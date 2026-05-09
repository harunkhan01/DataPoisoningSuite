"""
Register dataset class objects into a shared dictionary for easy instantiation

Register transforms into a shared dictionary for easy instantiation

"""
DATASET_REGISTRY = {}

def register_dataset(name):

    def wrapper(cls):
        DATASET_REGISTRY[name] = cls
        return cls

    return wrapper


TRANSFORM_REGISTRY = {}

def register_transform(name):

    def wrapper(func):
        TRANSFORM_REGISTRY[name] = func
        return func
    
    return wrapper