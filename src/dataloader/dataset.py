import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from cub import Cub

def get_dataset(dataset):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if dataset == "cub200":
        trans = transforms.Compose([
            transforms.RandomResizedCrop(224), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_trans = transforms.Compose([
            transforms.Resize(224), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            normalize,
        ])
        trainset = Cub(train=True, transform=trans, return_image_only=False)
        testset = Cub(train=False, transform=test_trans, return_image_only=False)
    elif dataset == "cifar10":
        trans = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=trans)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=trans)
    elif dataset == "oxford102":
        trans = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        trainset = torchvision.datasets.Flowers102(
            root='./data', split='train', download=True, transform=trans)
        testset = torchvision.datasets.Flowers102(
            root='./data', split='test', download=True, transform=trans)
    else:
        raise NotImplementedError(f"Dataset specified [{dataset}] not implemented")
    
    return trainset, testset

def get_transforms():
    # TODO: Figure out why transform for CUB is different in lime vs resnet training
    pil_transf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess_trans = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    return pil_transf, preprocess_trans