# Copyright2019 Gang Wei wg0502@bu.edu

import random
from copy import deepcopy
import numpy as np


def getCluster(X, U):
    """
    :param X: X is a set of x
    :param u: u is a set of u
    :return: [category,category,....]
    """
    error = 0
    allocate = [0] * len(X)
    # allocate = []
    # kk = 0
    for n in range(len(X)):   # Traverse all n points
        curr_min_d = 9999
        for k in range(len(U)):  # Traverse all k centers
            d = sum([np.sum(np.square(X[n][i] - U[k][i])) for i in range(len(U[k]))])
            # d = sum([getDistance(X[n][i], U[k][i]) for i in range(len(U[k]))])  # calculate d
            if d < curr_min_d:
                curr_min_d = d
                # kk = k
                allocate[n] = k
        # allocate.append(kk)
        error += curr_min_d
    return allocate, error


def iteration(data, U):
    """
    :param data: data is all x in X
    :return: allocation
    """
    # random chosen u
    Uprior = []
    Ucurr = deepcopy(U)
    # allocates = [0] * len(data)
    it = 0
    al, er = 0, 0
    while Ucurr != Uprior:
        it += 1
        print("iteration:", it)
        Uprior = deepcopy(Ucurr)
        [al, er] = getCluster(data, Ucurr)
        mean_d = 0

        for k in range(len(U)):
            Xk = []
            for i in range(len(al)):
                if al[i] == k:
                    Xk.append(data[i])
            Ucurr[k] = getmean(Xk)
            print(k, len(Xk))
            for f in range(len(Ucurr[k])):
                mean_d += getDistance(Ucurr[k][f], Uprior[k][f])
        if mean_d <= 0.01 or it >= 50:
            break

    return al, er


def getmean(X):
    """
    :param X: X is a set of x
    :return: the mean of set X
    """
    jnumber = len(X)  # number of points in it
    u = deepcopy(X[0])          # initialize u
    if len(u) == 0:
        return 0
    for i in range(len(u)):
        for j in range(len(u[i])):
            s = sum([X[point][i][j] for point in range(len(X))])
            avg = s / jnumber
            u[i][j] = avg
    return u


def getDistance(x1, x2):
    """
    x1,x2 should be same length
    x1,x2 are 1-d vectors
    :param x1: a vector
    :param x2: a vector
    :return: the 2-norm distance between x1 and x2
    """
    l1 = len(x1)
    l2 = len(x2)
    if l1 != l2:
        print('Error! Unequal vectors.')
        return False
    nx1 = np.array(x1)
    nx2 = np.array(x2)
    d = np.sum(abs(nx1 - nx2))
    '''d = 0
    for i in range(0,l1):
        d += (float(x1[i]) - float(x2[i]))**2'''
    d = d / l1
    return d


def randomData():
    """
    Just output random data for test
    :return: random data
    """
    XX = []
    for i in range(2000):
        X = []
        for k in range(1, 11):  # features
            x = []
            for j in range(0, k):
                x.append(random.randint(0, 1))
            X.append(x)
        XX.append(X)
    for i in range(2000):
        X = []
        """for k in range(1, 6):  # features
            x = []
            for j in range(0, k):
                x.append(random.randint(0, 1))
            X.append(x)"""
        for k in range(1, 11):  # features
            x = []
            for j in range(0, k):
                x.append(1)
            X.append(x)
        XX.append(X)
    return XX


# if __name__ == '__main__':
    # data = randomData()
    # number = len(data)
    # print(iteration(data))
    # print(getmean(data))


    # print(getDistance(x1[0],x2[0]))

