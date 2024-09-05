import json
from pathlib import Path
from typing import Dict, List

import pyrallis
import torch

import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from concept_embedding_training.data_utils import EmbeddingData
from concept_heads.clip.head import CLIPConceptHead
from concept_heads.face_recognition.head import FaceConceptHead
from configs.inference_config import InferenceConfig
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
def main(cfg: InferenceConfig):
    seed_everything(cfg.seed)

    # Load the concept head that was previously trained
    if cfg.concept_type == ConceptType.OBJECT:
        head_path = cfg.concept_head_path / f'{CLIP_MODEL_NAME}-{cfg.concept_name}-{cfg.head_data_type}-{cfg.n_head_positive_samples}-{cfg.n_head_negative_samples}-step-{cfg.classifier_step}.pt'
        concept_head = CLIPConceptHead(head_path)
    else:
        concept_head = FaceConceptHead()

    concept_signals = concept_head.extract_signal(cfg.image_paths)
    concept_signals = [concept_signals[path] for path in cfg.image_paths]

    vlm_wrapper = VLM_TYPE_TO_WRAPPER[cfg.vlm_type](device=cfg.device, torch_dtype=cfg.torch_dtype)
    myvlm = VLM_TYPE_TO_MYVLM[cfg.vlm_type](vlm=vlm_wrapper,
                                            layer=VLM_TO_LAYER[cfg.vlm_type],
                                            concept_name=cfg.concept_name,
                                            cfg=cfg)

    if cfg.personalization_task == 'recognition':
        iteration_to_concept_data = torch.load(cfg.checkpoint_path /
                                           f'concept_embeddings_{cfg.vlm_type}_vqa-{cfg.n_concept_embedding}-{cfg.head_data_type}-{cfg.n_head_positive_samples}-{cfg.n_head_negative_samples}.pt')
    else :
        iteration_to_concept_data = torch.load(cfg.checkpoint_path /
                                           f'concept_embeddings_{cfg.vlm_type}_{cfg.personalization_task}-{cfg.n_concept_embedding}-{cfg.head_data_type}-{cfg.n_head_positive_samples}-{cfg.n_head_negative_samples}.pt')

    # what is this?
    # epoch 같은 느낌인듯([25, 50, 75, 99]로 되어 있음)
    # print(iteration_to_concept_data.keys())
    # print(iteration_to_concept_data[25]['values'].size())
    # exit()

    iterations = cfg.iterations if cfg.iterations is not None else list(iteration_to_concept_data.keys())

    outputs = run_inference(myvlm=myvlm,
                            concept_signals=concept_signals,
                            iterations=iterations,
                            iteration_to_concept_data=iteration_to_concept_data,
                            cfg=cfg)

    # Save results to json file
    with open(cfg.inference_output_path / f'inference_outputs_{cfg.vlm_type}_{cfg.personalization_task}-{cfg.n_concept_embedding}-{cfg.head_data_type}-{cfg.n_head_positive_samples}-{cfg.n_head_negative_samples}.json', 'w') as f:
        json.dump(outputs, f, indent=4)


def run_inference(myvlm: MyVLM,
                  concept_signals: Dict[str, torch.Tensor],
                  iterations: List[int],
                  iteration_to_concept_data: Dict[str, EmbeddingData],
                  cfg: InferenceConfig) -> Dict[str, Dict]:

    # print(cfg.image_paths)
    # exit()

    print("*" * 100)
    print("RUNNING INFERENCE")
    print("*" * 100)
    outputs = {}
    for iteration in iterations:
        print('#' * 100)
        print(f"Running on iteration: {iteration}")
        inference_utils.set_concept_embeddings(vlm_wrapper=myvlm.vlm,
                                               concept_embeddings=iteration_to_concept_data[iteration],
                                               iteration=iteration,
                                               cfg=cfg)
        print('-' * 100)
        outputs[f'iteration_{iteration}'] = {}
        for image_idx, image in enumerate(cfg.image_paths):
            outputs[f'iteration_{iteration}'][str(image)] = {}
            for prompt in cfg.prompts:
                prompt = prompt.format(concept=cfg.concept_identifier)  # Add the identifier to prompt, if needed
                inputs = myvlm.vlm.preprocess(image, prompt)
                output = myvlm.vlm.generate(inputs, concept_signals=concept_signals[image_idx])
                outputs[f'iteration_{iteration}'][str(image)][prompt] = output[0]
                print(f"{Path(image).stem} | Input: {prompt} | Output: {output[0]}")

                if cfg.vlm_type == VLMType.MINIGPT_V2 and '[refer]' in prompt:
                    output_path = cfg.inference_output_path / 'rec_results' / \
                                  f'{Path(image).stem}---iteration_{iteration}.jpg'
                    myvlm.vlm.draw_and_save_referring_localization_result(image_tensor=inputs['image'],
                                                                          result_str=output[0],
                                                                          output_path=output_path)

            print('-' * 100)
        print("#" * 100)
    return outputs


if __name__ == '__main__':
    main()
