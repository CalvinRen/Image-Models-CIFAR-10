import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.dataset import train_dataloader
import os
from config.config import args
import wandb
from model import ResNet18
import numpy as np
import timm


# device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 卷积加速
torch.backends.cudnn.benchmark = True


# wandb init
if args.wandb:
    wandb.init(project='vision_models', name='{}_trainning'.format(args.model))
    wandb.config.update(args)


#load dataset
train_loader, valid_loader = train_dataloader(args)


# load model
if args.model == 'resnet18':
    model = ResNet18().to(device)
elif args.model == 'vit_timm':
    model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
    model.head = nn.Linear(768, 10).to(device)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    for name, param in model.named_parameters():
        if 'attn' in name:
            param.requires_grad = True
else:
    raise NotImplementedError


# loss and optim
criterion = nn.CrossEntropyLoss()
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
else:
    raise NotImplementedError

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)


# train
for epoch in tqdm(range(args.epoch)):
    correct = 0
    total = 0
    # train
    model.train()
    for i, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(output, 1)
        total += label.size(0)
        correct += (pred == label).sum().item()

        train_acc = correct / total
        wandb.log({'train_loss': loss.item(), 'train_acc': train_acc})
    
    # valid 
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, (image, label) in enumerate(valid_loader):
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            loss = criterion(output, label)
            _, pred = torch.max(output, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
            
            valid_acc = correct / total
            wandb.log({'val_loss': loss.item(), 'val_acc': valid_acc})

    # scheduler.step()

# save model
torch.save(model.state_dict(), './checkpoints/{}.pth'.format(args.model))

# finish wandb
wandb.finish()