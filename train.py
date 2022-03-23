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
            np.savez_compressed(self.save_path + 'experiment/parameter_' + str(self.cfg['pca_dim']) + '_' + str(
                    self.cfg['d']) + '_' + str(self.cfg['rho'])[2:], parameter=parameter)
        else:
            # np.save(self.save_path + 'experiment\\a0', a0)
            np.savez_compressed(
                self.save_path + 'experiment\\parameter_' + str(self.cfg['pca_dim']) + '_' + str(
                    self.cfg['d']) + '_' + str(self.cfg['rho'])[2:], parameter=parameter)
        print("save parameters. end.")


if __name__ == '__main__':
    config_path = 'config/train_config.yml'
    if_remote = True
    config = load_config(config_path, if_remote=if_remote)
    trainer = Trainer(config, if_remote)
    trainer.train()

# czt up 0.005  down 0.6
# czt1 up unknow  down 0.001
# czt2 up 0.01  down
