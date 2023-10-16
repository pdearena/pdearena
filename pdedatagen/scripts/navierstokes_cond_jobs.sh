#!/bin/bash
# This script produces 6656 training, 1664 valid, and 1664 test trajectories of the conditioned NS dataset.
seeds=(197910 452145 540788 61649 337293 296323 471551 238734 \
795028 806339 144207 415274 950547 391920 891493 645944 \
431652 391355 600690 495919 874847 97373 403121 588319 \
991570 597761 453345 349940 666497 597535 61891 975433 \
856942 788627 234361 433043 153164 126188 946091 795833 \
901348 142003 515976 509478 857366 766053 792079 585498 \
772145 954313 429673 445536)
for SEED in ${seeds[*]};
do
    python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \
    experiment=smoke_cond mode=train samples=128 seed=$SEED \
    dirname=pdearena_data/navierstokes_cond
    python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \
    experiment=smoke_cond mode=valid samples=32 seed=$SEED \
    dirname=pdearena_data/navierstokes_cond
    python scripts/generate_data.py base=pdedatagen/configs/navierstokes2dsmoke.yaml \
    experiment=smoke_cond mode=test samples=32 seed=$SEED \
    dirname=pdearena_data/navierstokes_cond
done
