#!/usr/bin/env bash
for ratio in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
python3 reference_low_rank.py \
    --ratio ${ratio} \
    --arch 'resnet56' \
    --batch-size 128 \
    --workers 1
done

for ratio in 0.19 0.31 0.465 
do
python3 reference_low_rank.py \
    --ratio ${ratio} \
    --arch 'densenet40' \
    --batch-size 128 \
    --workers 1
done
