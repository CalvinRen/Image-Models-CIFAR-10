CUDA_VISIBLE_DEVICES=1 python test.py --model='resnet18' \
                                    --batch_size=128 \
                                    --load_model_path='./checkpoints/resnet18.pth'