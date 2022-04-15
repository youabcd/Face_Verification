import numpy as np


# 余弦相似度
def cosine_similarity(x, y, a):
    ax = np.dot(a, x.T)
    ay = np.dot(a, y.T)
    ax_norm = np.linalg.norm(ax, axis=0)
    ay_norm = np.linalg.norm(ay, axis=0)
    return np.diagonal(np.dot(ax.T, ay)) / (ax_norm * ay_norm)


# 余弦相似度梯度
def grad_cs(x, y, a):
    ax = np.dot(a, x)
    ay = np.dot(a, y)
    ax_norm = np.linalg.norm(ax)
    ay_norm = np.linalg.norm(ay)
    ua = np.dot(ax, ay)
    # grad_u = np.dot(a, (2 * np.dot(x, y)))
    grad_u = a @ (x[:, np.newaxis] * y + y[:, np.newaxis] * x)
    va = ax_norm * ay_norm
    grad_v = ((ay_norm / ax_norm) * ax[:, np.newaxis]) * x - ((ax_norm / ay_norm) * ay[:, np.newaxis]) * y
    return grad_u / va - (ua / np.square(va)) * grad_v


def max_cs(data):
    idx = np.where(data < 0)[0]
    data[idx] = 0
    return data, idx


class ObjFunc1(object):
    def __init__(self, pos, neg, t1, t2, a0):
        self.pos = pos
        self.neg = neg
        self.t1 = t1
        self.t2 = t2
        self.a0 = a0

    def function(self, a):
        a = a.reshape(self.a0.shape)
        pos_sum = cosine_similarity(x=self.pos[:, 0, :], y=self.pos[:, 1, :], a=a)
        neg_sum = cosine_similarity(x=self.neg[:, 0, :], y=self.neg[:, 1, :], a=a)
        pos_sum, _ = max_cs(self.t1 - pos_sum)
        neg_sum, _ = max_cs(neg_sum - self.t2)
        return pos_sum.sum() + neg_sum.sum()

    def grad(self, a):
        a = a.reshape(self.a0.shape)
        pos_cs = cosine_similarity(x=self.pos[:, 0, :], y=self.pos[:, 1, :], a=a)
        neg_cs = cosine_similarity(x=self.neg[:, 0, :], y=self.neg[:, 1, :], a=a)
        _, pos_idx = max_cs(self.t1 - pos_cs)
        _, neg_idx = max_cs(neg_cs - self.t2)
        pos_sum = np.zeros(self.a0.shape)
        neg_sum = np.zeros(self.a0.shape)
        for i in range(len(self.pos)):
            if i not in pos_idx:
                pos_sum = pos_sum - grad_cs(x=self.pos[i][0], y=self.pos[i][1], a=a)
        for i in range(len(self.neg)):
            if i not in neg_idx:
                neg_sum = neg_sum + grad_cs(x=self.neg[i][0], y=self.neg[i][1], a=a)
        return (pos_sum + neg_sum).reshape(-1)


class ObjFunc2(object):
    def __init__(self, pos, neg, a0, alpha, beta):
        self.pos = pos
        self.neg = neg
        self.a0 = a0
        self.alpha = alpha
        self.beta = beta

    def function(self, a):
        a = a.reshape(self.a0.shape)
        pos_sum = cosine_similarity(x=self.pos[:, 0, :], y=self.pos[:, 1, :], a=a)
        neg_sum = cosine_similarity(x=self.neg[:, 0, :], y=self.neg[:, 1, :], a=a)
        pos_sum = pos_sum.sum()
        neg_sum = neg_sum.sum()
        return -(pos_sum - (self.alpha * neg_sum) - (self.beta * np.square(np.linalg.norm(a - self.a0))))

    def grad(self, a):
        a = a.reshape(self.a0.shape)
        pos_sum = np.zeros(self.a0.shape)
        neg_sum = np.zeros(self.a0.shape)
        for i in range(len(self.pos)):
            pos_sum = pos_sum + grad_cs(x=self.pos[i][0], y=self.pos[i][1], a=a)
        for i in range(len(self.neg)):
            neg_sum = neg_sum + grad_cs(x=self.neg[i][0], y=self.neg[i][1], a=a)
        return -((pos_sum - (self.alpha * neg_sum) - (2 * self.beta * (a - self.a0))).reshape(-1))


def grad_cs_pal(x, y, a):
    ax = np.einsum('ij, kj->ik', a, x)
    ay = np.einsum('ij, kj->ik', a, y)
    ax_norm = np.linalg.norm(ax, axis=0)
    ay_norm = np.linalg.norm(ay, axis=0)
    ua = np.einsum('ij,ij->j', ax, ay)
    va = ax_norm * ay_norm
    a1 = a.reshape((1, a.shape[0], a.shape[1]))
    xy = np.einsum('ij,ij->i', x, y)
    grad_u_va = a1 * (2 * xy / va).reshape([-1, 1, 1])
    test1 = np.einsum('ij, jk->jik', ax, x)
    test2 = ((ay_norm / ax_norm) * (ua / np.square(va))).reshape((-1, 1, 1))
    test3 = test1 * test2
    test4 = ((ay_norm / ax_norm) * (ua / np.square(va))).reshape((-1, 1, 1)) * np.einsum('ij, jk->jik', ax, x)
    grad_v_ua = ((ay_norm / ax_norm) * (ua / np.square(va))).reshape((-1, 1, 1)) * np.einsum('ij, jk->jik', ax, x) - (
            (ax_norm / ay_norm) * (ua / np.square(va))).reshape((-1, 1, 1)) * np.einsum('ij, jk->jik', ay, y)
    return grad_u_va - grad_v_ua


# def grad_func(pos, neg, a, a0, alpha, beta):
#     pos_all = grad_cs_pal(x=pos[:, 0, :], y=pos[:, 1, :], a=a)
#     neg_all = grad_cs_pal(x=neg[:, 0, :], y=neg[:, 1, :], a=a)
#     pos_sum = np.sum(pos_all, axis=0)
#     neg_sum = np.sum(neg_all, axis=0)
#     return pos_sum - (alpha * neg_sum) - (2 * beta * (a - a0))
