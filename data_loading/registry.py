"""
Register dataset class objects into 

"""
DATASET_REGISTRY = {}

def register_dataset(name):

    def wrapper(cls):
        DATASET_REGISTRY[name] = cls
        return cls

    return wrapper