import numpy as np
from data_set import local_bp
from utils.pca import PCA


def train_data_loader(path, size):
    x_data = np.zeros([len(path), size])
    for i in range(len(path)):
        img = local_bp(path[i])
        x_data[i:i+1] = img[:size]
    pca = PCA(x_data, n_components=size - 1000)
    y = pca.reduce_dimension()
    return y


if __name__ == '__main__':
    train_data_loader(['../face_data/temp/face1.jpg', '../face_data/temp/face2.jpg'], 8000)
