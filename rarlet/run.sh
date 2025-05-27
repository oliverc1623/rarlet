#!/bin/bash
# Execute scripts with different seeds and additional arguments for torchcompile scripts
scripts=(
    sac_metadrive_protagonist.py
)
for script in "${scripts[@]}"; do
    for seed in 31 41 51; do
        if [[ $script == *_protagonist.py ]]; then
            python $script \
                --seed=$seed \
                --exp_name "lane-follow" \
                --num-envs 8 \
                --gradient_steps -1 \
                --cudagraphs \
                --compile \
                --map "SSS" \
                --num-lanes 2 \
                --no-random_spawn_lane_index \
                --total-timesteps 500_000
        else
            python $script --seed=$seed
        fi
    done
done
