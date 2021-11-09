import numpy as np


def logistic_regression(data, rate):
    x = data[:, :3]

    
    w = np.zeros(data.shape[1] -1)
    direction = np.zeros(data.shape[1] -1)
    for _ in range(500):
        for n in range(data.shape[0]):
            negativescore = -data[n, 3] * np.dot(w, data[n, :3])
            theta = 1 / (1 + np.exp(-negativescore))
            direction += theta * -data[n, 3] * data[n, :3]
        # direction can be viewed as gradient
        direction /= data.shape[0]
        # 把正負號處理好後會發現，更新理念其實跟PLA一樣，都是朝score大的方向去更新
        w -= rate * direction
    return w
