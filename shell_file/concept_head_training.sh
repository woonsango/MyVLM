#!/bin/bash

# python3 ./example_configs/concept_head_training/write_yaml.py

# 특정 디렉토리 설정
DIRECTORY="./dataset/data_concept_head/positives/"

# 파일 이름들을 배열에 저장
concepts=($(ls "$DIRECTORY"))

# 배열의 파일 이름들을 하나씩 출력하는 for문
for concept in "${concepts[@]}"
do
    CUDA_VISIBLE_DEVICES=6 python3 ./concept_heads/clip/concept_head_training/train.py --config_path example_configs/concept_head_training/concept/"${concept}"_concept_head_training.yaml
done