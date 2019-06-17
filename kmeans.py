# Copyright2019 Gang Wei wg0502@bu.edu

import random
from copy import deepcopy
import numpy as np
import time
from numpy.random import choice


def getCluster(X, U):
    """
    :param X: X is a set of x
    :param u: u is a set of u
    :return: [category,category,....]
    """
    error = 0

    # allocate = [0] * len(X)
    x_in_k = {}
    for i in range(len(U)):
        x_in_k[i] = []
    for n in range(len(X)):   # Traverse all n points
        curr_min_d = 9999
        closest = '#'
        for k in range(len(U)):  # Traverse all k centers
            # d = np.sum([np.sum(np.square(np.array(X[n][i]) - np.array(U[k][i]))) for i in range(len(U[k]))])
            d = sum([getDistance(X[n][i], U[k][i]) for i in range(len(U[k]))])  # calculate d
            if d < curr_min_d:
                curr_min_d = d
                closest = k
                # allocate[n] = k
        x_in_k[closest].append(X[n])
        error += curr_min_d
    error /= len(X[0])
    for i in range(len(U)):
        if not x_in_k[i]:
            x_in_k[i].append(x_in_k[0].pop())
    return x_in_k, error
    # return allocate, error


def iteration(data, U):
    """
    :param data: data is all x in X
    :return: allocation
    """
    Uprior = []
    Ucurr = deepcopy(U)
    it = 0
    while True:
        it += 1
        print("iteration:", it)
        Uprior = deepcopy(Ucurr)
        dict, er = getCluster(data, Ucurr)
        mean_d = 0
        for k in range(len(U)):
            '''Xk = []
            for i in range(len(al)):
                if al[i] == k:
                    Xk.append(data[i])
            Ucurr[k] = getmean(Xk)
            print(k, len(Xk))'''
            Ucurr[k] = getmean(dict[k])
            print(k, len(dict[k]))
            for f in range(len(Ucurr[k])):
                mean_d += getDistance(Ucurr[k][f], Uprior[k][f])
        if mean_d <= 0.01 or it >= 50:
            break
    return dict, er
    # return al, er


def getmean(X):
    """
    :param X: X is a set of x
    :return: the mean of set X
    """
    if not X:
        print("empty X")
    # jnumber = len(X)  # number of points in it

    u = deepcopy(X[0])          # initialize u
    if len(u) == 0:
        return 0
    for i in range(len(u)):
        if type(u[i])==list:
            for j in range(len(u[i])):
                s = sum([X[point][i][j] for point in range(len(X))])
                avg = s / len(X)
                u[i][j] = avg
        else:
            s = sum([X[point][i] for point in range(len(X))])
            avg = s / len(X)
            u[i] = avg

    return u


def getDistance(x1, x2):
    """
    x1,x2 should be same length
    x1,x2 are 1-d vectors
    :param x1: a vector
    :param x2: a vector
    :return: the 2-norm distance between x1 and x2
    """
    if type(x1)==list:
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
            d += (float(x1[i]) - float(x2[i]))**2
        d = d / l1'''
        return d
    else:
        return abs(x1 - x2)


def get_k_means_pp(X, U):
    Ufix = deepcopy(U)
    for k in range(len(U)-1):
        d = [0] * len(X)
        indx = [i for i in range(len(X))]
        for n in range(len(X)):
            s = sum([getDistance(X[n][i], Ufix[k][i]) for i in range(len(X[n]))])
            d[n] = np.square(s)
        d_sum = sum(d)
        for n in range(len(X)):
            d[n] /= d_sum
        ch = choice(indx, p=d)
        Ufix[k+1] = X[ch]
    return Ufix


# if __name__ == '__main__':
    # data = randomData()
    # number = len(data)
    # print(iteration(data))
    # print(getmean(data))


