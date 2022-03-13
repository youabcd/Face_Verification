import numpy as np
import yaml
from utils.cosine_similarity_metric_learning import cs_ml


class Trainer(object):
    def __init__(self, t):
        self.t = t

    def train(self):
        return cs_ml(pos=self.t)


if __name__ == '__main__':
    trainer = Trainer(1)
    best_a = trainer.train()
