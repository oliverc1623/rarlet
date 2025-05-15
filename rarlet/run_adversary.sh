#!/bin/bash
# Execute scripts with different seeds and additional arguments for torchcompile scripts
scripts=(
    sac_metadrive_adversary.py
)
for script in "${scripts[@]}"; do
    for n in 1 2 5 8; do
        if [[ $script == *_adversary.py ]]; then
            python $script --seed=1 --exp-name "adversary0" --env-id "SSSS" --num-envs 8 --gradient-steps -1 --num_idm_vehicles 1 --compile --cudagraphs
        else
            python $script --seed=$seed
        fi
    done
done
