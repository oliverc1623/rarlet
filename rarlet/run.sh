#!/bin/bash
# Execute scripts with different seeds and additional arguments for torchcompile scripts
scripts=(
    sac_metadrive_protagonist.py
)
for script in "${scripts[@]}"; do
    for seed in 21 31 41 51; do
        if [[ $script == *_protagonist.py ]]; then
            python $script --seed=$seed --exp_name "SAC_protagonist" --num-envs 8 --gradient_steps -1 --cudagraphs --compile
        else
            python $script --seed=$seed
        fi
    done
done
