import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import os

import torch

from myvlm.common import ConceptType, VLMType, PersonalizationTask, VLM_TO_PROMPTS, VALID_IMAGE_EXTENSIONS, TrainDataType


@dataclass
class RecognitionAbilityConfig:
    # The identifier that was used for concept_embedding_training the concept
    concept_identifier: str
    # Which type of concept is this? Person or object?
    concept_type: ConceptType
    # Which VLM you are running inference on
    vlm_type: VLMType
    # Which personalization task you want to run inference on
    train_personalization_task: PersonalizationTask
    # List of image paths we wish to run inference on. If a path is given, we iterate over this directory
    image_paths: Union[Path, List[str]]

    n_concept: int

    # Where are the concept embedding checkpoints saved to? This should contain a directory for each concept.
    checkpoint_path: Path = Path('./outputs')
    # Where to save the results to
    output_path: Path = Path('./eval/recognition_ability/outputs')
    # Where are the linear heads for the object concepts saved? This should contain a directory for each concept.
    concept_head_path: Path = Path('./object_concept_heads')
    # Which step of the concept head to use if working with objects
    classifier_step: int = 500
    # Defines which seed to use for the concept head and for the concept embeddings
    seed: int = 42
    # Which iterations to run inference on. If None, will run on all iterations that were saved
    iterations: Optional[List[int]] = None
    # List of prompts to use for inference. If None, we will use the default set of prompts defined in `common/common.py`
    prompts: Optional[List[str]] = None
    # Device to train on
    device: str = 'cuda'
    # Torch dtype
    torch_dtype: torch.dtype = torch.bfloat16

    n_head_positive_samples: int = 4
    n_head_negative_samples: int = 4
    head_data_type: str = 'base'
    n_concept_embedding: int = 1

    concept_list: List[str] = None

    train_data_type: TrainDataType = TrainDataType.HEAD

    negative_recognition: bool = True

    def __post_init__(self):

        # Get the prompts. If None is given, then we use the default list for each VLM and task
        self.negative_prompts = None
        if self.prompts is None:
            self.prompts = VLM_TO_PROMPTS[self.vlm_type].get('recognition', None)
            print(self.prompts)
            # exit()
            if self.prompts is None:
                raise ValueError(f"Prompts for task 'Recognition' are not defined for {self.vlm_type}!")
            print(self.negative_recognition)
            if self.negative_recognition :
                self.negative_prompts = VLM_TO_PROMPTS[self.vlm_type].get('negativeRecognition', None)

                print(self.negative_prompts)

        if self.concept_list is None:
            self.concept_list = os.listdir(self.image_paths / 'positives')
        print('eval concept list : ',self.concept_list)
        print(type(self.concept_list))
        # exit()

        if not self.concept_head_path.name.startswith('seed_'):
            concept_head_paths = {}
            for concept_name in self.concept_list:
                concept_head_paths[concept_name] = self.concept_head_path / concept_name / f'seed_{self.seed}'
            self.concept_head_path = concept_head_paths

        print(self.concept_head_path)
        # exit()

        if not self.checkpoint_path.name.startswith('seed_'):
            checkpoint_paths = {}
            for concept_name in self.concept_list:
                checkpoint_paths[concept_name] = self.checkpoint_path / concept_name / f'seed_{self.seed}'
            self.checkpoint_path = checkpoint_paths
        print(self.checkpoint_path)
        # exit()

        self._verify_concept_embeddings_exist()
        if self.concept_type == ConceptType.OBJECT:
            self._verify_concept_heads_exist()

        # exit()

        self.eval_inference_output_path = self.output_path / 'eval_inference_outputs' / f'seed_{self.seed}'
        self.eval_inference_output_path.mkdir(parents=True, exist_ok=True)
        
        self.positives_image_paths = {}
        for concept in self.concept_list:
            self.positives_image_paths[concept] = self.image_paths / 'positives' / concept

        self.negatives_image_paths = self.image_paths / 'random'

        print(self.positives_image_paths)
        print(self.negatives_image_paths)

        # exit()

        for concept, positives_image_path in self.positives_image_paths.items():
            if type(positives_image_path) == pathlib.PosixPath and positives_image_path.is_dir():
                self.positives_image_paths[concept] = [str(p) for p in positives_image_path.glob('*') if p.suffix in VALID_IMAGE_EXTENSIONS]
                with open(self.checkpoint_path[concept] / f'train_paths_{concept}-{self.vlm_type}_{self.train_personalization_task}-{self.n_concept_embedding}-{self.train_data_type}-{self.head_data_type}-{self.n_head_positive_samples}-{self.n_head_negative_samples}.txt', 'r') as f:
                    concept_embedding_train_paths = [str(positives_image_path/Path(path.strip('\n')).name) for path in list(f)]
                
                self.positives_image_paths[concept] = list(set(self.positives_image_paths[concept]) - set(concept_embedding_train_paths))

        print(self.positives_image_paths)
        # exit()

        if type(self.negatives_image_paths) == pathlib.PosixPath and self.negatives_image_paths.is_dir():
            self.negatives_image_paths = [str(p) for p in self.negatives_image_paths.glob('*') if p.suffix in VALID_IMAGE_EXTENSIONS]

        print(self.negatives_image_paths)
        # exit()

        # Set the threshold value for recognizing the concept
        self.threshold = 0.5 if self.concept_type == ConceptType.OBJECT else 0.675

    def _verify_concept_heads_exist(self):
        for concept_head_path in self.concept_head_path.values() :
            if not concept_head_path.exists() and self.concept_type == ConceptType.OBJECT:
                raise ValueError(f"Concept head path {concept_head_path} does not exist!")

    def _verify_concept_embeddings_exist(self):
        for checkpoint_path in self.checkpoint_path.values():
            if not checkpoint_path.exists():
                raise ValueError(f"Concept checkpoint path {checkpoint_path} does not exist!")
