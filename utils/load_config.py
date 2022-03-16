import yaml
from addict import Dict


def load_config(path, if_remote=True):
    if if_remote:
        cfg = Dict(yaml.safe_load(open(path)))
    else:
        cfg = Dict(yaml.safe_load(open(path, encoding='utf-8')))
    if cfg['ap'] == 'WPCA' and cfg['if_regenerate'] is False:
        raise RuntimeError('A_WPCA need regenerate data')
    if cfg['d'] > cfg['pca_dim']:
        raise ValueError('d must lower than pca_dim')
    return cfg
