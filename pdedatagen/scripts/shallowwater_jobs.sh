#!/bin/bash
# This script produces 5.6k training, 1.4k valid, and 1.4k test trajectories of the Shallow water dataset.
seeds=(197910 452145 540788 61649 337293 296323 471551 238734 \
795028 806339 144207 415274 950547 391920 891493 645944 \
431652 391355 600690 495919 874847 97373 403121 588319 \
991570 597761 453345 349940 666497 597535 61891 975433 \
856942 788627 234361 433043 153164 126188 946091 795833 \
901348 142003 515976 509478 857366 766053 792079 585498 \
772145 954313 429673 445536 799432 146142 19024 438811)
for SEED in ${seeds[*]};
do
    python scripts/generate_data.py base=pdedatagen/configs/shallowwater.yaml \
    experiment=shallowwater mode=train samples=100 seed=$SEED \
    dirname=pdearena_data/shallowwater;
    python scripts/generate_data.py base=pdedatagen/configs/shallowwater.yaml \
    experiment=shallowwater mode=valid samples=25 seed=$SEED \
    dirname=pdearena_data/shallowwater;
    python scripts/generate_data.py base=pdedatagen/configs/shallowwater.yaml \
    experiment=shallowwater mode=test samples=25 seed=$SEED \
    dirname=pdearena_data/shallowwater;
done

for mode in train valid test; do
    python scripts/convertnc2zarr.py "pdearena_data/shallowwater/$mode";
done

python scripts/compute_normalization.py --dataset shallowwater pdearena_data/shallowwater
