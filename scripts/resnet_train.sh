CUDA_VISIBLE_DEVICES=3 python resnet_train.py --model=resnet18 \
                                        --batch_size=128 \
                                        --lr=0.1 \
                                        --optimizer=sgd \
                                        --epoch=200 

