#!/bin/bash
# Execute scripts with different seeds and additional arguments for torchcompile scripts
scripts=(
    sac_metadrive_protagonist.py
)
traffic_densities=(0.0 0.2 0.3)
for script in "${scripts[@]}"; do
    for seed in 31 41 51; do
        if [[ $script == *_protagonist.py ]]; then
            for density in "${traffic_densities[@]}"; do
                python $script --seed=$seed \
                    --exp_name "lane-follow" \
                    --num-envs 4 \
                    --gradient_steps -1 \
                    --cudagraphs \
                    --compile \
                    --map "SSS" \
                    --num-lanes 2 \
                    --no-random_spawn_lane_index \
                    --total-timesteps 1000 \
                    --alpha 0.1 \
                    --traffic-density $density
            done
        else
            python $script --seed=$seed
        fi
    done
done
