CUDA_VISIBLE_DEVICES=3 python test.py --model='vit_timm' \
                                    --batch_size=128 \
                                    --load_model_path='./checkpoints/vit.pth'