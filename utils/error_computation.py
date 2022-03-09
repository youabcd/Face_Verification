import numpy as np


# test data
# t = [[[1,2,3,4,5],[5,4,3,2,1],0],[[4,1,3,2,5],[4,2,3,2,5],1],[[1,2,3,4,5],[1,2,3,4,4],1],[[4,1,3,2,5],[5,2,4,1,3],0]]
def cs(data):
    ans = []
    for i in range(len(data)):
        x_norm = np.linalg.norm(data[i][0])
        y_norm = np.linalg.norm(data[i][1])
        ans.append(np.dot(data[i][0], data[i][1]) / (x_norm * y_norm))
    return ans


def compute_error(t, a, k):
    for p in t:
        p[0] = np.dot(p[0], a)
        p[1] = np.dot(p[1], a)
    total_error = 0
    t_split = np.array_split(t, k)
    cos_sim = []
    for i in range(k):
        cos_sim.append(cs(t_split[i]))
    for i in range(k):
        pri_theta = 1.
        max_cnt = 0
        test_error = 0
        for theta in np.arange(0.5, 1, 0.01):
            cnt = 0
            for j in range(k):
                if j != i:
                    for sub in range(len(t_split[j])):
                        if (cos_sim[j][sub] > theta and t_split[j][sub][2] == 1) or (
                                cos_sim[j][sub] <= theta and t_split[j][sub][2] == 0):
                            cnt += 1
            if cnt > max_cnt:
                max_cnt = cnt
                pri_theta = theta
        print(pri_theta, max_cnt)
        for sub in range(len(t_split[i])):
            if (cos_sim[i][sub] <= pri_theta and t_split[j][sub][2] == 1) or (
                    cos_sim[j][sub] > pri_theta and t_split[j][sub][2] == 0):
                test_error += 1
        print(test_error)
        total_error = total_error + test_error
    return total_error / k
