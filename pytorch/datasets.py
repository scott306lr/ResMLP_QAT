from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from timm.data import create_dataset
from torch.utils.data import DataLoader
import torch, os

def data_loader(name, root, input_size, batch_size, num_workers=0):
    dataset_train, NUM_CLASSES = build_dataset(is_train=True, name=name, root=root, input_size=input_size, is_tfds=False)
    dataset_val, _ = build_dataset(is_train=False, name=name, root=root, input_size=input_size, is_tfds=False)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * batch_size),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return data_loader_train, data_loader_val, NUM_CLASSES

def tfds_data_loader(name, root, input_size, batch_size, num_workers=0):
    dataset_train, NUM_CLASSES = build_dataset(is_train=True, name=name, root=root, input_size=input_size, is_tfds=True)
    dataset_val, _ = build_dataset(is_train=False, name=name, root=root, input_size=input_size, is_tfds=True)

    data_loader_train = DataLoader(
        dataset_train, #sampler=sampler_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    data_loader_val = DataLoader(
        dataset_val, #sampler=sampler_val,
        batch_size=int(1.5 * batch_size),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return data_loader_train, data_loader_val, NUM_CLASSES

def build_dataset(is_train, name, root, input_size, is_tfds=False):
    transform = build_transform(is_train, input_size)

    if is_tfds:
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
    
    else:
        if name == 'cifar10':
            dataset = datasets.CIFAR100(root=root, train=is_train, transform=transform)
            nb_classes = 10
        elif name == 'imagenet2012':
            root = os.path.join(root, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root=root, transform=transform)
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