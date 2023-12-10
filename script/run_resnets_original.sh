#!/usr/bin/env bash
mkdir -p results

# Reference Network training
for arch in resnet56
do
python3 reference_trainer.py --arch ${arch} \
        --batch-size 128 --scheduler steps --milestones 100 150 \
        --lr 0.1 --lr_decay 0.1 --momentum 0.9 --epochs 200 \
        --checkpoint 20 --print-freq 5 \
        --dataset CIFAR10 | tee -a references/${arch}_reference.log
done
