import json
import pyrallis

import matplotlib.pyplot as plt

from pathlib import Path
from typing import List

import numpy as np

import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from myvlm.common import seed_everything, VLMType, PersonalizationTask, TrainDataType

from dataclasses import dataclass

@dataclass
class RecognitionVisualizationConfig:
    vlm_type: VLMType                         # Can also be LLAVA or MINIGPT_V2
    train_personalization_task: PersonalizationTask        # Can also be VQA or REC depending on the VLM type
    seed: int = 42 # This should be the same seed used for the concept head (for objects) and embedding training

    output_path: Path = Path('./eval/recognition_ability/outputs')

    n_head_positive_samples: int = 4
    n_head_negative_samples: int = 4
    head_data_type: str = 'base'
    n_concept_embedding: int = 1

    train_data_type: TrainDataType = TrainDataType.HEAD # or RANDOM (random is random choice, head is train data that is concept_head training data and same about test data)

    concept_list: List[str] = None

    def __post_init__(self):

        self.eval_inference_output_path = self.output_path / 'eval_inference_outputs' / f'seed_{self.seed}'

        self.n_concept = len(self.concept_list)


def load_json(cfg : RecognitionVisualizationConfig):
    with open(cfg.eval_inference_output_path / f'inference_outputs_{cfg.vlm_type}_recognition-{cfg.n_concept}-{cfg.n_concept_embedding}-{cfg.head_data_type}-{cfg.n_head_positive_samples}-{cfg.n_head_negative_samples}.json') as json_file:
        json_data = json.load(json_file)

    return json_data

# 기존에 acc를 안 구했던 json에 acc를 더해주는 함수
def preprocessing(cfg: RecognitionVisualizationConfig, json_data):

    for index in range(len(json_data['result'])):
        for key, value in json_data['result'][index]['question'].items():
            # print(json_data['result'][index]['question'][key])   
            if 'positives' in json_data['result'][index]['question'][key].keys():     
                json_data['result'][index]['question'][key]['result'] = json_data['result'][index]['question'][key]['positives']/json_data['result'][index]['question'][key]['sum']
                json_data['result'][index]['question'][key].pop('reuslt',None)
            else:
                json_data['result'][index]['question'][key]['result'] = json_data['result'][index]['question'][key]['negatives']/json_data['result'][index]['question'][key]['sum']
                json_data['result'][index]['question'][key].pop('reuslt',None)

    with open(cfg.eval_inference_output_path / f'inference_outputs_{cfg.vlm_type}_recognition-{cfg.n_concept}-{cfg.n_concept_embedding}-{cfg.head_data_type}-{cfg.n_head_positive_samples}-{cfg.n_head_negative_samples}.json', 'w') as f:
        json.dump(json_data, f, indent=4)
    
    return json_data

def choice_question_result(json_data, index):

    json_data = json_data['result'][index]['question']
    
    question_result = [ (key, value['result']) for key, value in json_data.items()]

    return question_result

def order_best_qeustion(question_result):
    question_result.sort(key= lambda x : x[1])

    return question_result

    # positives qeustion 뽑기
    # negatives question 뽑기
    # not postivies qeustion 뽑기
    # not negatives question 뽑기

def draw_graph(positives_question_result, negatives_question_result):

    # 정수형 라벨과 값, 각 라벨에 대한 설명 정의
    
    labels = np.array(range(len(positives_question_result)))
    positives_values = np.array([q_r[1] for q_r in positives_question_result])
    negatives_values = np.array([q_r[1] for q_r in negatives_question_result])

    # 막대의 너비 설정
    bar_width = 0.5

    distance = abs(positives_values-negatives_values)

    indices = np.where(distance >= 0.4)[0]
    print(indices)
    
    high_distance = [positives_question_result[index][0] for index in indices]
    print(high_distance)

    with open('./eval/recognition_ability/visualization/high_distnace.txt', 'w') as f:
            for question in high_distance:
                f.write(f"{question}\n")


    fig, ax = plt.subplots(2)
    bars_distance = ax[1].bar(labels , distance, bar_width, color='lavender')

    # print(labels)
    # print(positives_values)
    # print(negatives_values)
    # exit()

    # # 막대 그래프 생성
    # fig, ax = plt.subplots()
    # bars = ax.bar(labels, positives_values, color='skyblue')

    bars1 = ax[0].bar(labels - bar_width / 2, positives_values, bar_width, color='skyblue')
    bars2 = ax[0].bar(labels + bar_width / 2, negatives_values, bar_width, color='salmon')

    # 그래프 라벨 추가
    ax[0].set_xlabel('Labels')
    ax[0].set_ylabel('Values')

    ax[0].legend((bars1[0], bars2[0]), ('postives', 'negatives'), fontsize=8)
    ax[1].set_ylabel('difference')

    plt.subplots_adjust(hspace=0.5)

    plt.savefig('./eval/recognition_ability/visualization/bar_graph.png')
    plt.show()

@pyrallis.wrap()
def main(cfg : RecognitionVisualizationConfig):
    seed_everything(cfg.seed)

    json_data = load_json(cfg)
    # json_data = preprocessing(cfg, json_data)

    postives_question_result = choice_question_result(json_data, 0)
    print(postives_question_result)
    negatives_question_result = choice_question_result(json_data, 1)
    print(negatives_question_result)


    draw_graph(postives_question_result, negatives_question_result)

    # 순서대로 나열하기
    # sorted_question_result = order_best_qeustion(question_result)
    # print(sorted_question_result)

    

if __name__ == '__main__':
    main()