# Vision Models on CIFAR-10
Computer Vision Models implemented in PyTorch. Train and Test on CIFAR-10 dataset. To be updated...

## TODO List
- [x] Auto-Encoder
- [x] Variational Auto-Encoder
- [x] ResNet 
- [x] Vision Transformer (ViT) 
- [ ] MAE (TODO)
- [ ] Swin Transformer (TODO)
- [ ] Diffusion Model (TODO)

## Results
| Model | Test Accuracy | Checkpoints | Training Log |
| :---: | :---: | :---: | :---: |
| ResNet-18 | 95.% | [Checkpoints](https://drive.google.com/file/d/1Uw6S46igmqtZ_tPJj8JCZ0mCHYrzhsyE/view?usp=sharing) | [Log](https://api.wandb.ai/links/1969347522/qvxyxgt3) |
| vit_base_patch16_224 | 98.87% | [Checkpoints](https://drive.google.com/file/d/1Uw6S46igmqtZ_tPJj8JCZ0mCHYrzhsyE/view?usp=sharing) | [Log](https://api.wandb.ai/links/1969347522/ihzb67uf) |

## ResNet-18
- Dataset Cutout
- Learning Rate Scheduler
- 7x7 Conv -> 3x3 Conv
- Remove MaxPool
- KaiMing_Normal for initialization


## Vision Transformer (ViT)
Used pre-trained checkpoints(vit_base_patch16_224) from [TIMM]((https://github.com/huggingface/pytorch-image-models)).


## Reference
1. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929v2)
3. [Github/kentaroy47/vision-transformers-cifar10](https://github.com/kentaroy47/vision-transformers-cifar10)
4. [pytorch-image-models](https://github.com/huggingface/pytorch-image-models)