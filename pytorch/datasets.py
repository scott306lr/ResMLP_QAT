from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from timm.data import create_dataset

def build_dataset(is_train, name, root, input_size):
    transform = build_transform(is_train, input_size)

    if name == 'cifar10':
        split = 'train' if is_train else 'test'
        dataset = create_dataset(name='tfds/cifar10', root=root, split=split, transform=transform)
        nb_classes = 10
    elif name == 'imagenet2012':
        split = 'train' if is_train else 'validation'
        dataset = create_dataset(name='tfds/imagenet2012', root=root, split=split, transform=transform)
        nb_classes = 1000
    else:
        raise NameError('name should be "cifar10" or "imagenet2012"')

    return dataset, nb_classes


def build_transform(is_train, input_size):
    if is_train:
        return get_train_transforms(input_size)
    else:
        return get_test_transforms_resmlp(input_size)
    
def get_train_transforms(input_size):
    transform = create_transform(
        input_size=input_size,
        is_training=True,
        color_jitter=0.3,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
    )
    return transform
    
def get_test_transforms_resmlp(input_size):
    mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD   
    transformations = {}
    Rs_size = int(input_size/0.9)
    transformations = transforms.Compose(
        [transforms.Resize(Rs_size, interpolation=3),
         transforms.CenterCrop(input_size),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    return transformations

def get_test_transforms_org(input_size):
    mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD   
    transformations = {}
    Rs_size = int((256 / 224) * input_size)
    transformations = transforms.Compose(
        [transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
         transforms.CenterCrop(input_size),
         transforms.ToTensor(),
         transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
    return transformations