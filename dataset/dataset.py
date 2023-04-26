import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset
from .Cutout import Cutout




# Train dataloader
def train_dataloader(args):
    if args.model == 'vit_timm':
        size = 224
    elif args.model == 'resnet18':
        size = 24
    else:
        raise NotImplementedError

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=1, length=16))

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = CIFAR10(root=args.dataset_path, train=True, download=False, transform=train_transform)
    valid_dataset = CIFAR10(root=args.dataset_path, train=True, download=False, transform=valid_transform)
    
    # 将训练集拆分成训练集和验证集
    train_indices = list(range(len(train_dataset)))
    train_split = int(len(train_dataset) * 0.8)
    valid_split = len(train_dataset) - train_split
    train_indices, valid_indices = torch.utils.data.random_split(train_indices, [train_split, valid_split])

    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(Subset(valid_dataset, valid_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, valid_loader


# Test dataloader
def test_dataloader(args):
    if args.model == 'vit_timm':
        size = 224
    elif args.model == 'resnet18':
        size = 24
    else:
        raise NotImplementedError
    
    test_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_dataset = CIFAR10(root=args.dataset_path, train=False, download=False, transform=test_transform)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return test_loader
