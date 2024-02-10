#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering
input_dir=../../data/raw

setting=20-100

clustering_name=anwar

d1=${setting}
d2=${clustering_name}
python ${source_dir}/perform_previous_clustering.py \
    --input_dev_file ${data_dir}/dataset/${d1}/exemplars_dev.jsonl \
    --input_test_file ${data_dir}/dataset/${d1}/exemplars_test.jsonl \
    --input_w2v_file ${input_dir}/word2vec/GoogleNews-vectors-negative300.bin \
    --output_dir ${data_dir}/previous_clustering/${d1}/${d2} \
    --clustering_name ${clustering_name}
