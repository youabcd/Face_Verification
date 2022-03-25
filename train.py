from utils.cosine_similarity_metric_learning import cs_ml
from utils.change_data import change_data, change_data_1
from utils.load_config import load_config
from utils.get_ap import get_ap
import numpy as np


class Trainer(object):
    def __init__(self, cfg, if_remote):
        self.cfg = cfg
        self.if_remote = if_remote
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
        a0_s, min_cve_s, best_theta, best_beta = cs_ml(pos=self.pos, neg=self.neg, t=self.t, d=self.d, ap=self.ap,
                                                       k=self.k, repeat=self.max_repeat, rho=self.cfg['rho'])
        print("finish training")
        parameter = dict()
        parameter['min_cve_s'] = min_cve_s
        parameter['best_theta'] = best_theta
        parameter['best_beta'] = best_beta
        parameter['a0_s'] = a0_s
        if self.if_remote is True:
            # np.save(self.save_path + 'experiment/a0', a0)
            np.savez_compressed(self.save_path + 'experiment/009/parameter_' + str(self.cfg['pca_dim']) + '_' + str(
                    self.cfg['d']), parameter=parameter)
        else:
            # np.save(self.save_path + 'experiment\\a0', a0)
            np.savez_compressed(
                self.save_path + 'experiment\\parameter_' + str(self.cfg['pca_dim']) + '_' + str(
                    self.cfg['d']) + '_fl', parameter=parameter)
        print("save parameters. end.")


if __name__ == '__main__':
    config_path = 'config/train_config.yml'
    if_remote = False
    config = load_config(config_path, if_remote=if_remote)
    trainer = Trainer(config, if_remote)
    trainer.train()

# czt  up 200,400 now_best    down 240,400 now_best
# czt1 up now_best    down now_best
# czt2 up now_best    down 50,100 now_best*
# czt3 up 30,50 now_best    down 25,50 now_best
# local now_best
