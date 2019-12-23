import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def loadData(filename):
    df = pd.read_csv(filename, sep='	', names=['1', '2', 'target'])
    data = []
    target = []
    for i in range(len(df)):
        d = df.iloc[i]
        ds = [d['1'], d['2']]
        t = int(d['target'])
        if t == 0:
            t = -1
        data.append(ds)
        target.append(t)
    return np.mat(data), target

if __name__ == '__main__':
    dataMatrix, target = loadData('../../data/testSetRBF2.txt')
    print(len(target), dataMatrix.shape, type(target))
    # print(dataMatrix)
    # print(target)