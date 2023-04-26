CUDA_VISIBLE_DEVICES=0 python train.py --model='vit_timm' \
                                        --batch_size=128 \
                                        --lr=0.001 \
                                        --optimizer='sgd' \
                                        --epoch=10