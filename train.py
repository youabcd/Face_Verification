import numpy as np
from utils.cosine_similarity_metric_learning import cs_ml
from utils.change_data import change_data
from utils.load_config import load_config
from utils.get_ap import get_ap


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.d = cfg['d']
        self.k = cfg['k']
        self.max_repeat = cfg['max_repeat']
        if cfg['if_regenerate'] is False:
            pos, neg = change_data(cfg['data_path'])
            self.pos = pos
            self.neg = neg
            self.t = ''
            self.ap = get_ap(cfg['ap'], cfg['d'], cfg['pca_dim'])

    def train(self):
        return cs_ml(pos=self.pos, neg=self.neg, t=self.t, d=self.d, ap=self.ap, k=self.k, repeat=self.max_repeat)


if __name__ == '__main__':
    config_path = 'config/train_config.yml'
    config = load_config(config_path)
    trainer = Trainer(config)
    best_a = trainer.train()
