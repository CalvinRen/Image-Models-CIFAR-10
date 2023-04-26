CUDA_VISIBLE_DEVICES=1 python ../resnet_train.py --model=resnet18 \
                                        --batch_size=128 \
                                        --lr=0.1 \
                                        --optimizer=sgd \
                                        --epoch=200 \
                                        --dataset_path=../data \
                                        --wandb_name='resnet18-new'

