import pyrallis

import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from concept_heads.clip.concept_head_training.coach import Coach
from concept_heads.clip.concept_head_training.config import ConceptHeadTrainingConfig


@pyrallis.wrap()
def main(cfg: ConceptHeadTrainingConfig):
    coach = Coach(cfg)
    coach.train()


if __name__ == '__main__':
    main()
