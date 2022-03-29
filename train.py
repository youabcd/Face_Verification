from utils.cosine_similarity_metric_learning import cs_ml
from utils.change_data import change_data
from utils.load_config import load_config
from utils.get_ap import get_ap
import numpy as np
from inference import compute_acc


class Trainer(object):
    def __init__(self, cfg, if_remote):
        self.cfg = cfg
        self.if_remote = if_remote
        self.d = cfg['d']
        self.k = cfg['k']
        self.max_repeat = cfg['max_repeat']
        self.save_path = cfg['save_path']
        if cfg['if_regenerate'] is False:
            pos, neg, t, _ = change_data(cfg['data_path'], cfg['pca_dim'])
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
                self.save_path + 'experiment\\gradient_descent\\parameter_' + str(self.cfg['pca_dim']) + '_' + str(
                    self.cfg['d']), parameter=parameter)
        print("save parameters. end.")
        return a0_s, min_cve_s, best_theta, best_beta


def test_result(path, pca_dim, d, ap, k, max_repeat, rho, repeat):
    print("a shape: ", ap.shape)
    total_acc = 0
    max_acc = 0
    min_acc = 1
    for i in range(repeat):
        pos, neg, t, test = change_data(path=path, dim=pca_dim)
        print(i)
        a0, min_cve, best_theta, best_beta = cs_ml(pos=pos, neg=neg, t=t, d=d, ap=ap, k=k, repeat=max_repeat, rho=rho)
        err, acc = compute_acc(test_data=test, theta=best_theta[0], a=a0[0])
        print("acc: ", acc)
        total_acc = total_acc + acc
        if acc > max_acc:
            max_acc = acc
        if acc < min_acc:
            min_acc = acc
    print("a shape: ", ap.shape)
    print("avg acc: ", total_acc / repeat)
    print("min acc: ", min_acc)
    print("max acc: ", max_acc)
    return total_acc, min_acc, max_acc


if __name__ == '__main__':
    config_path = 'config/train_config.yml'
    if_remote = True
    config = load_config(config_path, if_remote=if_remote)
    # trainer = Trainer(config, if_remote)
    # trainer.train()
    test_result(path=config['data_path'], pca_dim=config['pca_dim'], d=config['d'],
                ap=get_ap(config['ap'], config['d'], config['pca_dim']), k=config['k'], max_repeat=config['max_repeat'],
                rho=config['rho'], repeat=10)

# czt  up 100,200 now_best    down now_best
# czt1 up now_best    down now_best
# czt2 up now_best    down now_best
# czt3 up  now_best    down now_best
# local now_best
