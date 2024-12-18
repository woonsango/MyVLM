import json
from pathlib import Path
from typing import Dict, List

import pyrallis
import torch

import random

import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from concept_embedding_training.data_utils import EmbeddingData
from concept_heads.clip.head import CLIPConceptHead
from concept_heads.face_recognition.head import FaceConceptHead
from configs.eval_recognition_ability_config import RecognitionAbilityConfig
from inference import inference_utils
from myvlm import myblip2, myllava, myminigpt_v2
from myvlm.common import ConceptType, seed_everything, CLIP_MODEL_NAME, VLMType, VLM_TO_LAYER
from myvlm.myvlm import MyVLM
from vlms.blip2_wrapper import BLIP2Wrapper
from vlms.llava_wrapper import LLaVAWrapper
from vlms.minigpt_wrapper import MiniGPTWrapper

VLM_TYPE_TO_WRAPPER = {
    VLMType.BLIP2: BLIP2Wrapper,
    VLMType.LLAVA: LLaVAWrapper,
    VLMType.MINIGPT_V2: MiniGPTWrapper
}
VLM_TYPE_TO_MYVLM = {
    VLMType.BLIP2: myblip2.MyBLIP2,
    VLMType.LLAVA: myllava.MyLLaVA,
    VLMType.MINIGPT_V2: myminigpt_v2.MyMiniGPT_v2
}


@pyrallis.wrap()
def main(cfg: RecognitionAbilityConfig):
    seed_everything(cfg.seed)

    outputs = {}
    outputs['positives'] = {}
    outputs['negatives'] = {}
    positives_acc = {'all_positives':0, 'all_sum':0, 'concept':{}, 'question':{}}
    negatives_acc = {'all_negatives':0, 'all_sum':0, 'concept':{}, 'question':{}}

    negative_positives_acc = {'all_positives':0, 'all_sum':0, 'concept':{}, 'question':{}}
    negative_negatives_acc = {'all_negatives':0, 'all_sum':0, 'concept':{}, 'question':{}}

    for concept in cfg.concept_list:
        positives_acc['concept'][concept] = {'positives': 0, 'sum' : 0}
        negatives_acc['concept'][concept] = {'negatives': 0, 'sum' : 0}
        negative_positives_acc['concept'][concept] = {'positives': 0, 'sum' : 0}
        negative_negatives_acc['concept'][concept] = {'negatives': 0, 'sum' : 0}

    for question in cfg.prompts:
        question = question.format(concept=cfg.concept_identifier)
        positives_acc['question'][question] = {'positives': 0, 'sum' : 0}
        negatives_acc['question'][question] = {'negatives': 0, 'sum' : 0}
    
    if cfg.negative_recognition:
        for question in cfg.negative_prompts:
            question = question.format(concept=cfg.concept_identifier)
            negative_positives_acc['question'][question] = {'positives': 0, 'sum' : 0}
            negative_negatives_acc['question'][question] = {'negatives': 0, 'sum' : 0}

    print(positives_acc)
    print(negatives_acc)

    # Load the concept head that was previously trained
    if cfg.concept_type == ConceptType.OBJECT:
        head_path = {}
        concept_head = {}
        for concept_name in cfg.concept_list:
            head_path[concept_name] = cfg.concept_head_path[concept_name] / f'{CLIP_MODEL_NAME}-{concept_name}-{cfg.head_data_type}-{cfg.n_head_positive_samples}-{cfg.n_head_negative_samples}-step-{cfg.classifier_step}.pt'
            concept_head[concept_name] = CLIPConceptHead(head_path[concept_name])
    else:
        concept_head = FaceConceptHead()

    print(head_path)
    print(concept_head)
    # exit()

    positives_concept_signals = {}
    for concept_name in cfg.concept_list:
        positives_concept_signals[concept_name] = concept_head[concept_name].extract_signal(cfg.positives_image_paths[concept_name])
        # print(positives_concept_signals)
        # print(cfg.image_paths)
        positives_concept_signals[concept_name] = [positives_concept_signals[concept_name][path] for path in cfg.positives_image_paths[concept_name]]
    # print('*'*20)
    print(positives_concept_signals)
    # exit()

    # for key, value in positives_concept_signals.items():
        # print(key)
        # print(len(value))

    negatives_concept_signals = {}
    for concept_name in cfg.concept_list:
        negatives_concept_signals[concept_name] = concept_head[concept_name].extract_signal(cfg.negatives_image_paths)
        negatives_concept_signals[concept_name] = [negatives_concept_signals[concept_name][path] for path in cfg.negatives_image_paths]

    print(negatives_concept_signals)

    # for key, value in negatives_concept_signals.items():
        # print(key)
        # print(len(value))
    

    # exit()

    vlm_wrapper = VLM_TYPE_TO_WRAPPER[cfg.vlm_type](device=cfg.device, torch_dtype=cfg.torch_dtype)
    myvlm = VLM_TYPE_TO_MYVLM[cfg.vlm_type](vlm=vlm_wrapper,
                                            layer=VLM_TO_LAYER[cfg.vlm_type],
                                            concept_name= None,
                                            cfg=cfg)
    
    iteration_to_concept_data = {}
    for concept_name in cfg.concept_list:
        iteration_to_concept_data[concept_name] = torch.load(cfg.checkpoint_path[concept_name] /
                                        f'concept_embeddings_{cfg.vlm_type}_{cfg.train_personalization_task}-{cfg.n_concept_embedding}-{cfg.train_data_type}-{cfg.head_data_type}-{cfg.n_head_positive_samples}-{cfg.n_head_negative_samples}-{cfg.train_recognition_qeustion_answer}.pt')

    # print(iteration_to_concept_data)
    # for concept_name, values in iteration_to_concept_data.items():
    #     print(concept_name)
    #     for key, value in values.items():
    #         print(key)
    #         print(value['keys'].size())
    #         print(value['values'].size())
    # exit()

    # what is this?
    # epoch 같은 느낌인듯([25, 50, 75, 99]로 되어 있음)
    # print(iteration_to_concept_data.keys())
    # exit()

    iterations = {}
    for concept_name in cfg.concept_list:
        iterations[concept_name] = cfg.iterations if cfg.iterations is not None else list(iteration_to_concept_data[concept_name].keys())
    # print(iterations)
    # exit()

    for concept_name in cfg.concept_list:
        run_inference(myvlm=myvlm,
                        concept_signals=positives_concept_signals[concept_name],
                        iterations=iterations[concept_name],
                        iteration_to_concept_data=iteration_to_concept_data[concept_name],
                        concept_name = concept_name,
                        outputs = outputs['positives'],
                        acc = positives_acc,
                        negative_acc = negative_positives_acc,
                        image_type = 'positives',
                        image_paths = cfg.positives_image_paths,
                        cfg=cfg,
                        )
        run_inference(myvlm=myvlm,
                        concept_signals=negatives_concept_signals[concept_name],
                        iterations=iterations[concept_name],
                        iteration_to_concept_data=iteration_to_concept_data[concept_name],
                        concept_name = concept_name,
                        outputs = outputs['negatives'],
                        acc = negatives_acc,
                        negative_acc = negative_negatives_acc,
                        image_type = 'negatives',
                        image_paths = cfg.negatives_image_paths,
                        cfg=cfg,
                        )
        
    for concept_name in cfg.concept_list:
        positives_acc['concept'][concept_name]['result'] = positives_acc['concept'][concept_name]['positives']/positives_acc['concept'][concept_name]['sum']
        negatives_acc['concept'][concept_name]['result'] = negatives_acc['concept'][concept_name]['negatives']/negatives_acc['concept'][concept_name]['sum']
        if cfg.negative_recognition:
            negative_positives_acc['concept'][concept_name]['result'] = negative_positives_acc['concept'][concept_name]['positives']/negative_positives_acc['concept'][concept_name]['sum']
            negative_negatives_acc['concept'][concept_name]['result'] = negative_negatives_acc['concept'][concept_name]['negatives']/negative_negatives_acc['concept'][concept_name]['sum']

    for question in cfg.prompts:
        question = question.format(concept=cfg.concept_identifier)
        positives_acc['question'][question]['result'] = positives_acc['question'][question]['positives']/positives_acc['question'][question]['sum']
        negatives_acc['question'][question]['result'] = negatives_acc['question'][question]['negatives']/negatives_acc['question'][question]['sum']

    if cfg.negative_recognition:
        for question in cfg.negative_prompts:
            question = question.format(concept=cfg.concept_identifier)
            negative_positives_acc['question'][question]['result'] = negative_positives_acc['question'][question]['positives']/negative_positives_acc['question'][question]['sum']
            negative_negatives_acc['question'][question]['result'] = negative_negatives_acc['question'][question]['negatives']/negative_negatives_acc['question'][question]['sum']

    positives_acc['all_result'] = positives_acc['all_positives']/positives_acc['all_sum']
    negatives_acc['all_result'] = negatives_acc['all_negatives']/negatives_acc['all_sum']
    if cfg.negative_recognition:
        negative_positives_acc['all_result'] = negative_positives_acc['all_positives']/negative_positives_acc['all_sum']
        negative_negatives_acc['all_result'] = negative_negatives_acc['all_negatives']/negative_negatives_acc['all_sum']
    
    outputs['result'] = [positives_acc, negatives_acc]
    if cfg.negative_recognition:
        outputs['result'].append(negative_positives_acc)
        outputs['result'].append(negative_negatives_acc)
        
    # Save results to json file
    with open(cfg.eval_inference_output_path / f'inference_outputs_{cfg.vlm_type}_recognition-{cfg.n_concept}-{cfg.n_concept_embedding}-{cfg.head_data_type}-{cfg.n_head_positive_samples}-{cfg.n_head_negative_samples}-{cfg.train_recognition_qeustion_answer}.json', 'w') as f:
        json.dump(outputs, f, indent=4)
    print(positives_acc)
    print(negatives_acc)
    print(negative_positives_acc)
    print(negative_negatives_acc)


def run_inference(myvlm: MyVLM,
                  concept_signals: Dict[str, torch.Tensor],
                  iterations: List[int],
                  iteration_to_concept_data: Dict[str, EmbeddingData],
                  concept_name: str,
                  outputs: Dict,
                  acc: Dict,
                  negative_acc: Dict,
                  image_type: str,
                  image_paths: Dict,
                  cfg: RecognitionAbilityConfig) -> Dict[str, Dict]:

    # print(cfg.image_paths)
    # exit()
    outputs[concept_name] = {}
    print("*" * 100)
    print("RUNNING INFERENCE")
    print("*" * 100)
    for iteration in iterations:
        print('#' * 100)
        print(f"Running on iteration: {iteration}")
        inference_utils.set_concept_embeddings(vlm_wrapper=myvlm.vlm,
                                               concept_embeddings=iteration_to_concept_data[iteration],
                                               iteration=iteration,
                                               cfg=cfg)
        print('-' * 100)
        outputs[concept_name][f'iteration_{iteration}'] = {}
        for image_idx, image in enumerate(image_paths[concept_name] if image_type == 'positives' else image_paths):
            outputs[concept_name][f'iteration_{iteration}'][str(image)] = {}
            # local_prompts = random.sample(cfg.prompts, 6)
            local_prompts = cfg.prompts
            for prompt in local_prompts:
                prompt = prompt.format(concept=cfg.concept_identifier)  # Add the identifier to prompt, if needed
                inputs = myvlm.vlm.preprocess(image, prompt)
                output = myvlm.vlm.generate(inputs, concept_signals=concept_signals[image_idx])
                outputs[concept_name][f'iteration_{iteration}'][str(image)][prompt] = output[0]
                print(f"{Path(image).stem} | Input: {prompt} | Output: {output[0]}")
                acc['all_sum'] +=1
                acc['concept'][concept_name]['sum'] +=1
                acc['question'][prompt]['sum'] +=1
                if image_type == 'positives':
                    if 'Yes' in output[0]:
                        acc['all_positives'] += 1
                        acc['concept'][concept_name]['positives'] += 1
                        acc['question'][prompt]['positives'] += 1
                        print(f'acc[all_positives] : {acc["all_positives"]}, acc[all_sum] : {acc["all_sum"]} ({acc["all_positives"]/acc["all_sum"]}%)')
                        print(f'acc["concept"][{concept_name}]["positives"] : {acc["concept"][concept_name]["positives"]}, acc["concept"][{concept_name}]["sum"] : {acc["concept"][concept_name]["sum"]} ({acc["concept"][concept_name]["positives"]/acc["concept"][concept_name]["sum"]}%)')
                elif image_type == 'negatives':
                    if 'No' in output[0]:
                        acc['all_negatives'] += 1
                        acc["concept"][concept_name]["negatives"] += 1
                        acc["question"][prompt]["negatives"] += 1
                        print(f'acc[all_negatives] : {acc["all_negatives"]}, acc[all_sum] : {acc["all_sum"]} ({acc["all_negatives"]/acc["all_sum"]}%)')
                        print(f'acc["concept"][{concept_name}]["negatives"] : {acc["concept"][concept_name]["negatives"]}, acc["concept"][{concept_name}]["sum"] : {acc["concept"][concept_name]["sum"]} ({acc["concept"][concept_name]["negatives"]/acc["concept"][concept_name]["sum"]}%)')
                
                

                if cfg.vlm_type == VLMType.MINIGPT_V2 and '[refer]' in prompt:
                    output_path = cfg.inference_output_path / 'rec_results' / \
                                  f'{Path(image).stem}---iteration_{iteration}.jpg'
                    myvlm.vlm.draw_and_save_referring_localization_result(image_tensor=inputs['image'],
                                                                          result_str=output[0],
                                                                          output_path=output_path)
            if cfg.negative_recognition:
                local_prompts = cfg.negative_prompts
                for prompt in local_prompts:
                    
                    prompt = prompt.format(concept=cfg.concept_identifier)  # Add the identifier to prompt, if needed
                    inputs = myvlm.vlm.preprocess(image, prompt)
                    output = myvlm.vlm.generate(inputs, concept_signals=concept_signals[image_idx])
                    outputs[concept_name][f'iteration_{iteration}'][str(image)][prompt] = output[0]
                    print(f"{Path(image).stem} | Input: {prompt} | Output: {output[0]}")

                    negative_acc['all_sum'] +=1
                    negative_acc['concept'][concept_name]['sum'] +=1
                    negative_acc['question'][prompt]['sum'] +=1
                    
                    if image_type == 'positives':
                        if 'No' in output[0]:
                            negative_acc['all_positives'] += 1
                            negative_acc['concept'][concept_name]['positives'] += 1
                            negative_acc['question'][prompt]['positives'] += 1
                            print(f'negative_acc[all_positives] : {negative_acc["all_positives"]}, negative_acc[all_sum] : {negative_acc["all_sum"]} ({negative_acc["all_positives"]/negative_acc["all_sum"]}%)')
                            print(f'negative_acc["concept"][{concept_name}]["positives"] : {negative_acc["concept"][concept_name]["positives"]}, negative_acc["concept"][{concept_name}]["sum"] : {negative_acc["concept"][concept_name]["sum"]} ({negative_acc["concept"][concept_name]["positives"]/negative_acc["concept"][concept_name]["sum"]}%)')
                    elif image_type == 'negatives':
                        if 'Yes' in output[0]:
                            negative_acc['all_negatives'] += 1
                            negative_acc["concept"][concept_name]["negatives"] += 1
                            negative_acc["question"][prompt]["negatives"] += 1
                            print(f'negative_acc[all_negatives] : {negative_acc["all_negatives"]}, negative_acc[all_sum] : {negative_acc["all_sum"]} ({negative_acc["all_negatives"]/negative_acc["all_sum"]}%)')
                            print(f'negative_acc["concept"][{concept_name}]["negatives"] : {negative_acc["concept"][concept_name]["negatives"]}, negative_acc["concept"][{concept_name}]["sum"] : {negative_acc["concept"][concept_name]["sum"]} ({negative_acc["concept"][concept_name]["negatives"]/negative_acc["concept"][concept_name]["sum"]}%)')


            print('-' * 100)
        print("#" * 100)
    return outputs


if __name__ == '__main__':
    main()
