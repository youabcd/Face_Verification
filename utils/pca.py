import numpy as np


class PCA(object):
    def __init__(self, x, n_components=None):
        self.x = x
        self.dimension = x.shape[1]
        if n_components and n_components >= self.dimension:
            raise ValueError("n_components error")
        self.n_components = n_components

    def cov(self):
        x_t = np.transpose(self.x)
        means = x_t.mean(axis=1)
        means = means.reshape((-1, 1))
        x_t_c = x_t - means
        x_cov = np.cov(x_t_c)  # 协方差矩阵
        return x_cov

    def get_feature(self):
        x_cov = self.cov()
        a, b = np.linalg.eig(x_cov)  # 计算特征值和特征向量
        a = np.real(a)
        b = np.real(b)
        m = a.shape[0]
        c = np.hstack((a.reshape((m, 1)), b))
        sort = np.argsort(c[:, 0])
        return c[sort[::-1]], sort  # 按特征值从大到小排序

    def reduce_dimension(self):
        c_df_sort, sort = self.get_feature()
        p = c_df_sort[:self.n_components, 1:]
        features = c_df_sort[:self.n_components, 0]
        y = np.dot(p, np.transpose(self.x))
        return np.transpose(y), features, sort  # 返回降维后的结果


if __name__ == '__main__':
    # 例子 10个样本，每个样本3个特征
    # x = [[-6.1, -9.2, 9.2],
    #      [-1.1, 21.8, -6.8],
    #      [6.9, -3.2, 10.2],
    #      [-5.1, -15.2, 15.2],
    #      [25.9, 20.8, -8.8],
    #      [-7.1, 23.8, -14.8],
    #      [-5.1, -3.2, -5.8],
    #      [-8.1, -19.2, -4.8],
    #      [-5.1, -12.2, 1.2],
    #      [4.9, -4.2, 5.2],
    #      ]
    # add = [16.1, 24.2, 19.8]
    # add = np.array(add)
    # x = np.array(x)
    # x1 = x + add
    # pca = PCA(x, n_components=2)
    # y = pca.reduce_dimension()
    # print(y)
    data = np.load("/home/chenzhentao/fgfv_data/Intensity.npz", allow_pickle=True)['Intensity']
    pca = PCA(data, n_components=500)
    y, feature, sort = pca.reduce_dimension()
    sort = sort[::-1]
    sort = sort[:500]
    data = {
        'feature_value': feature,
        'feature_dim': sort
    }
    np.savez_compressed("/home/chenzhentao/fgfv_data/Intensity_pca_500.npz", pca=y)
    np.savez_compressed("/home/chenzhentao/fgfv_data/intensity_pca_feature_500.npz", feature=data)
