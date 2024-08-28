import os
import yaml
import copy

def check_concept_yaml() :

    dir_list = os.listdir('./example_configs/concept_head_training/concept')

    # if len(dir_list) == 0:
    #     return False
    # else :
    #     return True

    return dir_list

def write_concept_yaml() -> None :

    dir_list = check_concept_yaml()

    if len(dir_list) != 0 :
        print('ready concept_head_training yaml file')
        for yaml_file in dir_list:
            os.remove(f'./example_configs/concept_head_training/concept/{yaml_file}')

    with open('./example_configs/concept_head_training/base_concept_head_training.yaml') as f:
        base_training_yaml = yaml.load(f, Loader=yaml.FullLoader)

    # print(base_training_yaml)

    concept_name = os.listdir('./dataset/data_concept_head/positives')
    # print(concept_name)

    # training_yaml = []

    for i in concept_name:

        # base_training_yaml['concept_name'] = i
        # training_yaml.append(copy.deepcopy(base_training_yaml))
        concept_training_yaml = copy.deepcopy(base_training_yaml)
        concept_training_yaml['concept_name'] = i

        with open(f'./example_configs/concept_head_training/concept/{i}_concept_head_training.yaml', 'w') as f:
            yaml.dump(concept_training_yaml, f)
            

    # print(training_yaml)

        



        



if __name__ == '__main__':

    # print(check_concept_yaml())
    write_concept_yaml()

