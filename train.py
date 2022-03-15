from utils.cosine_similarity_metric_learning import cs_ml
from utils.change_data import change_data
from utils.load_config import load_config
from utils.get_ap import get_ap
import numpy as np


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.d = cfg['d']
        self.k = cfg['k']
        self.max_repeat = cfg['max_repeat']
        self.save_path = cfg['save_path']
        if cfg['if_regenerate'] is False:
            pos, neg, t = change_data(cfg['data_path'], cfg['pca_dim'])
            self.pos = pos
            self.neg = neg
            self.t = t
            self.ap = get_ap(cfg['ap'], cfg['d'], cfg['pca_dim'])
        else:
            img = cfg['image_path']

    def train(self):
        a0, min_cve_s, best_theta = cs_ml(pos=self.pos, neg=self.neg, t=self.t, d=self.d, ap=self.ap, k=self.k,
                                          repeat=self.max_repeat)
        print(min_cve_s)
        print(best_theta)
        print(a0)
        np.save(self.save_path + 'experiment/a0', a0)


if __name__ == '__main__':
    config_path = 'config/train_config.yml'
    config = load_config(config_path)
    trainer = Trainer(config)
    trainer.train()
