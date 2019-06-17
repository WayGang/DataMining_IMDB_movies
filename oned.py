
import random
import numpy as np
import kmeans
import csv
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import data_prepocess
import sys
sys.setrecursionlimit(999999999)


def A(l, m):

    if l+1 <= m:
        return 0
    if m == 1:
        return Union(0, l)

    r = []
    for j in range(m-1, l+1):
        r.append(A(j-1, m-1) + Union(j, l))
    re = min(r)
    index = r.index(re)
    s['%d'%l+','+'%d'%m] = index + m - 1
    return re


def Union(j,l):
    if l > j:
        return Union(j, l-1) + (((l-j) / (l-j+1)) * ((x[l] - mean(j, l-1))**2))
    else:
        return 0


def mean(j, l):
    if l > j:
        sum = 0
        for k in range(j, l+1):
            sum += x[k]
        return sum/(l-j+1)
    else:
        return x[j]


def run(k):
    global s
    s = {}
    x_ori = data_prepocess.pca_to_1d()
    x_ori = x_ori.tolist()
    for i in range(len(x_ori)):
        x_ori[i] = x_ori[i][0]
    global x
    x = sorted(x_ori)
    l = len(x) - 1
    m = k
    A(l,m)
    # print(s)
    left = []
    for i in range(m - 1):
        left.append(s['%d' % l + ',' + '%d' % m])
        l = s['%d' % l + ',' + '%d' % m] - 1
        m -= 1
    x_left = []
    for i in range(len(left)):
        x_left.append(x[left[i]])
    x_left.sort()
    assignments = [[i] for i in range(len(x_ori))]
    for i in range(len(x_ori)):
        for j in range(len(x_left)):
            if x_ori[i] < x_left[j]:
                assignments[i].append(j)
                break
        if x_ori[i] >= x_left[-1]:
            assignments[i].append(len(x_left)+1)
    # print(assignments)
    name = ['id', 'label']
    test = pd.DataFrame(columns=name, data=assignments)
    test.to_csv('./output_1d.csv')


def find_k_for_1d(k):
    errs = data_prepocess.find_k(k)
    kk = np.linspace(1, k, k)
    plt.plot(kk, errs, label='k-means')
    plt.show()



if __name__ == "__main__":
    find_k_for_1d(21)