import os
import random
from enum import Enum, auto

import numpy as np
import torch


class ConceptType(Enum):
    OBJECT = auto()
    PERSON = auto()


class VLMType(str, Enum):
    BLIP2 = 'blip-2'
    LLAVA = 'llava'
    MINIGPT_V2 = 'minigpt-v2'


class PersonalizationTask(str, Enum):
    CAPTIONING = 'captioning'
    VQA = 'vqa'
    REC = 'rec'
    Recognition = 'recognition'
    NegativeRecognition = 'negativeRecognition'


class MyVLMLayerMode(Enum):
    TRAIN = auto()
    INFERENCE = auto()

class TrainDataType(str, Enum):
    RANDOM = 'random'
    HEAD = 'head'


VLM_TO_LAYER = {
    VLMType.BLIP2: "model.vision_model.encoder.layers[38].mlp.fc2",
    VLMType.LLAVA: "model.model.mm_projector.linear2",
    VLMType.MINIGPT_V2: "model.llama_proj"
}

VLM_TO_EMBEDDING_DIM = {
    VLMType.BLIP2: 1408,
    VLMType.LLAVA: 4096,
    VLMType.MINIGPT_V2: 4096
}

VLM_TO_PROMPTS = {
    VLMType.BLIP2: {
        PersonalizationTask.CAPTIONING: [''],
    },
    VLMType.LLAVA: {
        PersonalizationTask.CAPTIONING: ['Please caption this image of {concept}.'],
        PersonalizationTask.VQA: [
            'Where is {concept} in the image?',
            'Where is {concept} positioned in the image?',
            'What is {concept} doing in the image?',
        ],
        PersonalizationTask.Recognition: [
            'Is {concept} in this photo? Answer briefly with yes or no.',
            'Can you tell if {concept} appears in this picture? Answer briefly with yes or no.',
            'Could you check whether {concept} is in the image? Answer briefly with yes or no.',
            'Do you see {concept} anywhere in this snapshot? Answer briefly with yes or no.',
            'Is there a chance {concept} could be in this photo? Answer briefly with yes or no.',
            'Would you happen to know if {concept} is shown in this photograph? Answer briefly with yes or no.',
            'Can you see {concept} in this photo? Answer briefly with yes or no.',
            'Have you spotted {concept} in this photo? Answer briefly with yes or no.',
            'Is that {concept} in the photo there? Answer briefly with yes or no.',
            'Is {concept} in this image? Answer briefly with yes or no.',
            'Am I seeing {concept} in this picture? Answer briefly with yes or no.',
            'Does this photo include {concept}? Answer briefly with yes or no.',
            'Is {concept} featured in this photo? Answer briefly with yes or no.',
            'Can you point out {concept} in this photo? Answer briefly with yes or no.',
            'Is {concept} visible in this photo? Answer briefly with yes or no.',
            'Check if {concept} is in this photo for me, will you? Answer briefly with yes or no.',
            'Hey AI, can you tell me if you see {concept} in this photo? Answer briefly with yes or no.',
            'Do you recognize {concept} in this photo? Answer briefly with yes or no.',
            'I’m looking for {concept}, is {concept} in this photo? Answer briefly with yes or no.',
            'Can you see if {concept} is in this photo? Answer briefly with yes or no.',
            'This photo, does it have {concept}? Answer briefly with yes or no.',
            'Could you confirm if this is {concept} in the photo? Answer briefly with yes or no.',
            'Any chance that {concept} might be in this photo? Answer briefly with yes or no.',
            'Can you recognize if {concept} is in this photo? Answer briefly with yes or no.',
            'I think I see {concept}, is it so? Answer briefly with yes or no.',
            'Has {concept} been captured in this photo? Answer briefly with yes or no.',
            '{concept}’s in this photo, right? Answer briefly with yes or no.',
            'Is {concept} present in this particular photo? Answer briefly with yes or no.',
            'Hey AI, can you tell me if you recognize {concept} in this photo? Answer briefly with yes or no.',
            ' Can you see if {concept} is in this photo? Answer briefly with yes or no.',
        ],
        PersonalizationTask.NegativeRecognition: [
            'Is {concept} not in this photo? Answer briefly with yes or no.',
            'Can you tell if {concept} doesn’t appear in this picture? Answer briefly with yes or no.',
            'Could you check whether {concept} is missing from the image? Answer briefly with yes or no.',
            'Do you not see {concept} anywhere in this snapshot? Answer briefly with yes or no.',
            'Is there no chance {concept} could be in this photo? Answer briefly with yes or no.',
            'Would you happen to know if {concept} is not shown in this photograph? Answer briefly with yes or no.',
            'Can’t you see {concept} in this photo? Answer briefly with yes or no.',
            'Have you not spotted {concept} in this photo? Answer briefly with yes or no.',
            'Is that not {concept} in the photo there? Answer briefly with yes or no.',
            'Is {concept} not in this image? Answer briefly with yes or no.',
            'Am I not seeing {concept} in this picture? Answer briefly with yes or no.',
            'Does this photo not include {concept}? Answer briefly with yes or no.',
            'Is {concept} not featured in this photo? Answer briefly with yes or no.',
            'Can’t you point out {concept} in this photo? Answer briefly with yes or no.',
            'Is {concept} not visible in this photo? Answer briefly with yes or no.',
            'Check if {concept} isn’t in this photo for me, will you? Answer briefly with yes or no.',
            'Hey AI, can you tell me if you don’t see {concept} in this photo? Answer briefly with yes or no.',
            'Do you not recognize {concept} in this photo? Answer briefly with yes or no.',
            'I’m looking for {concept}; isn’t {concept} in this photo? Answer briefly with yes or no.',
            'Can’t you see if {concept} is in this photo? Answer briefly with yes or no.',
            'This photo, doesn’t it have {concept}? Answer briefly with yes or no.',
            'Could you confirm if this is not {concept} in the photo? Answer briefly with yes or no.',
            'Any chance that {concept} might not be in this photo? Answer briefly with yes or no.',
            'Can’t you recognize if {concept} is in this photo? Answer briefly with yes or no.',
            'I think I don’t see {concept}, is it so? Answer briefly with yes or no.',
            'Has {concept} not been captured in this photo? Answer briefly with yes or no.',
            '{concept} isn’t in this photo, right? Answer briefly with yes or no.',
            'Is {concept} not present in this particular photo? Answer briefly with yes or no.',
            'Hey AI, can you tell me if you don’t recognize {concept} in this photo? Answer briefly with yes or no.',
            'Can’t you see if {concept} isn’t in this photo? Answer briefly with yes or no.',
        ]
    },
    VLMType.MINIGPT_V2: {
        PersonalizationTask.CAPTIONING: [
            '[caption] A short image caption of {concept}:',
            '[refer] {concept} in the image'
        ],
    }
}

VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

CLIP_MODEL_NAME = "DFN5B-CLIP-ViT-H-14-384"
MINIGPT_V2_CKPT_PATH = "/path/to/minigptv2_checkpoint.pth"
HF_TOKEN_FOR_LLAMA = 'IF WORKING WITH MINIGPT_V2, ENTER YOUR HF TOKEN FOR DOWNLOAIDNG LLAMA WEIGHTS'


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
