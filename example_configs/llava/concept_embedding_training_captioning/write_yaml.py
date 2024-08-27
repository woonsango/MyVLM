import os
import yaml
import copy

def check_concept_yaml() :

    dir_list = os.listdir('./example_configs/llava/concept_embedding_training_captioning')

    return dir_list

def write_concept_yaml() -> None :

    dir_list = check_concept_yaml()

    print(dir_list)

    # if len(dir_list) != 0 :
    #     print('ready concept_head_training yaml file')
    #     for yaml_file in dir_list:
    #         os.remove(f'./example_configs/concept_head_training/concept/{yaml_file}')

    # with open('./example_configs/concept_head_training/base_concept_head_training.yaml') as f:
    #     base_training_yaml = yaml.load(f, Loader=yaml.FullLoader)

    # concept_name = os.listdir('./dataset/data_concept_head/positives')

    # for i in concept_name:

    #     concept_training_yaml = copy.deepcopy(base_training_yaml)
    #     concept_training_yaml['concept_name'] = i

    #     with open(f'./example_configs/concept_head_training/concept/{i}_concept_head_training.yaml', 'w') as f:
    #         yaml.dump(concept_training_yaml, f)

if __name__ == '__main__':

    write_concept_yaml()

