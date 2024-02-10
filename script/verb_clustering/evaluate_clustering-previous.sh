#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering

setting=20-100

clustering_names=(arefyev anwar ribeiro)

splits=(dev test)

for clustering_name in ${clustering_names[@]}; do
    for split in ${splits[@]}; do
        d1=${setting}
        d2=${clustering_name}
        python ${source_dir}/evaluate_clustering.py \
            --input_file ${data_dir}/previous_clustering/${d1}/${d2}/exemplars_${split}.jsonl \
            --input_params_file ${data_dir}/previous_clustering/${d1}/${d2}/params.json \
            --output_dir ${data_dir}/evaluate_clustering_previous/${d1}/${d2} \
            --split ${split}
    done
done
