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



# wandb init
if args.wandb:
    wandb.init(project='vision_models', name=args.wandb_name)
    wandb.config.update(args)


#load dataset
train_loader, valid_loader = train_dataloader(args)


# load model
if args.model == 'resnet18':
    model = ResNet18().to(device)
elif args.model == 'vit_timm':
    model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
    model.head = nn.Linear(768, 10).to(device)
else:
    raise NotImplementedError


# train
counter = 0
valid_loss_min = np.Inf
lr = args.lr
for epoch in tqdm(range(args.epoch)):
    correct = 0
    total = 0
    valid_loss = 0.0
    # loss and optim
    if counter/10 ==1:
        counter = 0
        lr = lr*0.5
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise NotImplementedError

    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=args.epoch-1)

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
        wandb.log({'loss': loss.item(), 'acc': train_acc})
    
    # valid
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (image, label) in enumerate(valid_loader):
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            loss = criterion(output, label)
            valid_loss += loss.item()*image.size(0)
            _, pred = torch.max(output, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
            
            valid_acc = correct / total
            wandb.log({'val_acc': valid_acc})

    valid_loss = valid_loss / len(valid_loader.dataset)
    if valid_loss <= valid_loss_min:
        counter = 0
        valid_loss_min = valid_loss
    else:
        counter += 1


# save model
torch.save(model.state_dict(), './checkpoints/{}.pth'.format(args.model))

# finish wandb
wandb.finish()

