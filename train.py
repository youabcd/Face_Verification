from utils.cosine_similarity_metric_learning import cs_ml
from utils.change_data import change_data
from utils.load_config import load_config
from utils.get_ap import get_ap
import numpy as np
from inference import compute_acc
import time


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

    def train(self, if_save=True):
        a0_s, min_cve_s, best_theta, best_beta = cs_ml(pos=self.pos, neg=self.neg, t=self.t, d=self.d, ap=self.ap,
                                                       k=self.k, repeat=self.max_repeat, rho=self.cfg['rho'])
        print("finish training")
        parameter = dict()
        parameter['min_cve_s'] = min_cve_s
        parameter['best_theta'] = best_theta
        parameter['best_beta'] = best_beta
        parameter['a0_s'] = a0_s
        if if_save:
            if self.if_remote is True:
                # np.save(self.save_path + 'experiment/a0', a0)
                np.savez_compressed(self.save_path + 'experiment/cg/parameter_' + str(self.cfg['pca_dim']) + '_' + str(
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
    max_acc = 0
    min_acc = 1
    final_a = 0
    final_theta = 1.
    best_t1 = 0
    best_t2 = 0
    acc_res = {"t1": [], "t2": [], "acc": []}
    for t1 in np.linspace(0.5, 0.95, 10):
        for t2 in np.arange(-0.95, t1 + 0.01, 0.05):
            if t2 > t1:
                t2 = t1
            total_acc = 0
            for i in range(repeat):
                pos, neg, t, test = change_data(path=path, dim=pca_dim)
                print(i)
                a0, min_cve, best_theta, best_beta = cs_ml(pos=pos, neg=neg, t=t, d=d, ap=ap, k=k, repeat=max_repeat,
                                                           rho=rho, t1=t1, t2=t2)
                err, acc, same_acc, twin_acc = compute_acc(test_data=test, theta=best_theta[0], a=a0[0])
                print("acc: ", acc)
                print("same acc: ", same_acc)
                print("twin acc: ", twin_acc)
                print("theta: ", best_theta[0])
                total_acc = total_acc + acc
                if acc > max_acc:
                    max_acc = acc
                    final_a = a0
                    final_theta = best_theta
                    best_t1 = t1
                    best_t2 = t2
                if acc < min_acc:
                    min_acc = acc
            avg_acc = total_acc / repeat
            acc_res["t1"].append(t1)
            acc_res["t2"].append(t2)
            acc_res["acc"].append(avg_acc)
            print("avg acc: ", avg_acc)
    print("a shape: ", ap.shape)
    print("min acc: ", min_acc)
    print("max acc: ", max_acc)
    print("best t1: ", best_t1)
    print("best t2: ", best_t2)
    data = {
        'a0_s': final_a,
        'best_theta': final_theta
    }
    np.savez_compressed("/home/chenzhentao/Face_Verification/experiment/new_parameter_200_100.npz", parameter=data)
    np.savez_compressed("/home/chenzhentao/Face_Verification/experiment/t1_t2_test.npz", test=acc_res)
    return min_acc, max_acc


if __name__ == '__main__':
    config_path = 'config/train_config.yml'
    if_remote = True
    config = load_config(config_path, if_remote=if_remote)
    print("config: ", config)
    # trainer = Trainer(config, if_remote)
    # trainer.train()
    test_result(path=config['data_path'], pca_dim=config['pca_dim'], d=config['d'],
                ap=get_ap(config['ap'], config['d'], config['pca_dim']), k=config['k'], max_repeat=config['max_repeat'],
                rho=config['rho'], repeat=15)

# czt  up now_best    down now_best
# czt1 up now_best    down now_best
# czt2 up now_best    down now_best
# czt3 up now_best    down now_best
# local now_best
