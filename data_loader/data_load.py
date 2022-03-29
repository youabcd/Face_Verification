import os
import numpy as np
from data_set import local_bp, intensity
import sys

sys.path.append(os.path.split(sys.path[0])[0])
from utils.pca import PCA


def train_data_loader(pos_path, neg_path):
    file_list1 = os.listdir(pos_path)
    file_list2 = os.listdir(neg_path)
    file_list2.sort(key=lambda x: int(x[:-6]) * 10 + int(x[-5:-4]) if x.endswith('.jpg') else 0)
    path = []
    for i in range(len(file_list1)):
        if file_list1[i].endswith('.jpg'):
            path.append(pos_path + '/' + file_list1[i])
    for i in range(len(file_list2)):
        if file_list2[i].endswith('.jpg'):
            path.append(neg_path + '/' + file_list2[i])
    x_data = []
    path = np.array(path)
    for i in range(len(path)):
        if i % 100 == 0:
            print(i)
        # img = local_bp(path[i])
        img = intensity(path[i])
        x_data.append(img)
    x_data = np.array(x_data)
    np.savez_compressed("/home/chenzhentao/fgfv_data/Intensity.npz", Intensity=x_data)
    # pca = PCA(x_data, n_components=500)
    # y = pca.reduce_dimension()
    # return y


if __name__ == '__main__':
    train_data_loader("/home/chenzhentao/fgfv_data/aligned/same/", "/home/chenzhentao/fgfv_data/aligned/twins/")
