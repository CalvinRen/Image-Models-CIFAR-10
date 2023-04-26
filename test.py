import torch
import torch.nn as nn
from config.config import args
import os
from dataset.dataset import test_dataloader
from model import ResNet18
import timm


def test(args):
    # load dataset
    test_loader = test_dataloader(args)
    # load model
    if args.model == 'resnet18':
        model = ResNet18().to(device)
    elif args.model == 'vit_timm':
        model = timm.create_model('vit_base_patch16_224').to(device)
        model.head = nn.Linear(768, 10).to(device)
    else:
        raise NotImplementedError
    
    # load checkpoint
    model.load_state_dict(torch.load(args.load_model_path))

    # test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            _, pred = torch.max(output, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()

    print('Accuracy: {}'.format(correct / total))

if __name__ == '__main__':
    # device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test(args)
