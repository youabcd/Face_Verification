import numpy as np
from utils.error_computation import compute_error


def cosine_similarity(x, y, a):
    # x = [1,2,3,3,2]
    # y = [4,5,6,5,6]
    # a = [[1,2,3,1,2],[4,5,6,4,5]] m = 500 and d = 200
    ax = np.dot(a, x)
    ay = np.dot(a, y)
    ax_norm = np.linalg.norm(ax)
    ay_norm = np.linalg.norm(ay)
    return np.dot(ax, ay) / (ax_norm * ay_norm)


def grad_cs(x, y, a):
    ax = np.dot(a, x)
    ay = np.dot(a, y)
    ax_norm = np.linalg.norm(ax)
    ay_norm = np.linalg.norm(ay)
    ua = np.dot(ax, ay)
    grad_u = np.dot(a, (2 * np.dot(x, y)))
    va = ax_norm * ay_norm
    grad_v = ((ay_norm / ax_norm) * ax[:, np.newaxis]) * x - ((ax_norm / ay_norm) * ay[:, np.newaxis]) * y
    return grad_u / va - (ua / np.square(va)) * grad_v


def obj_func(pos, neg, a, a0, alpha, beta):
    pos_sum = 0
    neg_sum = 0
    for i in len(pos):
        pos_sum = pos_sum + cosine_similarity(x=pos[i][0], y=pos[i][1], a=a)
    for i in len(neg):
        neg_sum = neg_sum + cosine_similarity(x=neg[i][0], y=neg[i][1], a=a)
    return pos_sum - (alpha * neg_sum) - (beta * np.square(np.linalg.norm(a - a0)))


def grad_func(pos, neg, a, a0, alpha, beta):
    pos_sum = np.zeros(a0.shape)
    neg_sum = np.zeros(a0.shape)
    for i in len(pos):
        pos_sum = pos_sum + grad_cs(x=pos[i][0], y=pos[i][1], a=a)
    for i in len(neg):
        neg_sum = neg_sum + grad_cs(x=neg[i][0], y=neg[i][1], a=a)
    return pos_sum - (alpha * neg_sum) - (2 * beta * (a - a0))


def cg_arm(pos, neg, a0, alpha, beta):
    max_k = 5000
    rho = .6
    sigma = .4
    k = 0
    epsilon = 1e-4
    n = len(a0)
    a = a0
    g0 = np.ones(a0.shape)
    d0 = np.ones(a0.shape)
    while k < max_k:
        g = grad_func(pos=pos, neg=neg, a=a, a0=a0, alpha=alpha, beta=beta)
        item = k % n
        if item == 0:
            d = -g
        else:
            theta = np.linalg.norm(g, axis=1) / np.linalg.norm(g0, axis=1)
            d = -g + theta[:, np.newaxis] * d0
            # todo
            gd = g.T @ d
            if gd >= 0:
                d = -g
        if np.linalg.norm(g) < epsilon:
            break
        m = 0
        mk = 0
        while m < 20:
            if obj_func(pos=pos, neg=neg, a=(a + rho ** m * d), a0=a0, alpha=alpha, beta=beta) > obj_func(pos=pos,
                                                                                                          neg=neg, a=a,
                                                                                                          a0=a0,
                                                                                                          alpha=alpha,
                                                                                                          beta=beta) + sigma * rho ** m * g.T @ d:
                mk = m
                break
            m += 1
        a = a + rho ** mk * d
        g0 = g
        d0 = d
        k += 1
    return a


def cs_ml(pos, neg, t, d, ap, k, repeat):
    a0 = ap
    a_next = ap
    alpha = len(pos) / len(neg)
    for i in range(repeat):
        min_cve = np.finfo(np.float32).max
        for beta in np.arange(0.5, 5, 0.1):
            a1 = cg_arm(pos=pos, neg=neg, a0=a0, alpha=alpha, beta=beta)
            cve = compute_error(t=t, a=a1, k=k)
            if cve < min_cve:
                min_cve = cve
                a_next = a1
        a0 = a_next
