import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


def get_model(dataset, device, is_random=False, model_path=None):
    if is_random:
        model = resnet50()
    else:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = model.to(device)

    num_feats = model.fc.in_features

    if dataset == "cub200":
        # Don't Freeze the layers
        for param in model.parameters():
            param.requires_grad = True
        # Change last layer
        model.fc = nn.Sequential(nn.Linear(num_feats, 200) # 200 CUB categories
        )
    elif dataset == "cifar10":
        # Freeze the layers
        for param in model.parameters():
            param.requires_grad = False
        # Change last layer
        model.fc = nn.Sequential(
            nn.Linear(2048, 256), 
            nn.ReLU(), 
            nn.Linear(256, 10)  # 10 CIFAR classes
        )
    elif dataset == "oxford102":
        # Freeze the layers
        for param in model.parameters():
            param.requires_grad = False
        # Change last layer
        model.fc = nn.Sequential(nn.Linear(num_feats, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 102),   # 102 Oxford102 Flower categories
        )
    else:
        raise NotImplementedError(f"Dataset specified [{dataset}] not implemented")

    if model_path:
        model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    return model

def get_optimizer(model, params):
    optimizer = params.pop("optimizer")
    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), **params)
    elif optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), **params)
    elif optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), **params)
    else:
        raise NotImplementedError(f"Optimizer not found:[{params}]. Available optimizers: ['SGD', 'Adam', 'RMSprop']")

    return optimizer