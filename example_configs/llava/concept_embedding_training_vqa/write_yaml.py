import os
import yaml
import copy

def check_concept_yaml() :

    dir_list = os.listdir('./example_configs/llava/concept_embedding_training_vqa/concept')

    return dir_list

def write_concept_yaml() -> None :

    dir_list = check_concept_yaml()

    if len(dir_list) != 0 :
        print('ready concept_head_training yaml file')
        for yaml_file in dir_list:
            os.remove(f'./example_configs/llava/concept_embedding_training_vqa/concept/{yaml_file}')

    with open('./example_configs/llava/concept_embedding_training_vqa/base_concept_embedding_training_vqa.yaml') as f:
        base_training_yaml = yaml.load(f, Loader=yaml.FullLoader)

    concept_name = os.listdir('./dataset/data_concept_embedding/')

    for i in concept_name:

        concept_training_yaml = copy.deepcopy(base_training_yaml)
        concept_training_yaml['concept_name'] = i

        with open(f'./example_configs/llava/concept_embedding_training_vqa/concept/{i}_concept_embedding_training_vqa.yaml', 'w') as f:
            yaml.dump(concept_training_yaml, f)

if __name__ == '__main__':

    # print(check_concept_yaml())
    write_concept_yaml()

