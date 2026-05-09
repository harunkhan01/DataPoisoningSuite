from torchvision import transforms as T

from ..registry import register_transform

@register_transform('Grayscale')
def grayscale(num_output_channels=1):

    return T.Grayscale(num_output_channels)

@register_transform('RandomCrop')
def random_crop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):

    return T.RandomCrop(size, padding, pad_if_needed, fill, padding_mode)

@register_transform('RandomHorizontalFlip')
def random_horizontal_flip(p=0.5):

    return T.RandomHorizontalFlip(p)