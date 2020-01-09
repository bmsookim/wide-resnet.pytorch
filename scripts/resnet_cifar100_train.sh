#!/bin/bash
export netType='resnet'
export depth=18
export dataset='cifar100'

python main.py \
    --lr 0.1 \
    --net_type ${netType} \
    --depth ${depth} \
    --dropout 0 \
    --dataset ${dataset}
