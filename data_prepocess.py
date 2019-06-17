# Copyright2019 Gang Wei wg0502@bu.edu

import random
import numpy as np
import kmeans
import csv
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def read_file(path):
    """

    :param path: the location of movies.csv
    :return: [0] is features, [1] is set X
    """
    f = open(path, 'r')
    cf = csv.reader(f)
    X = []
    for row in cf:
        X.append(row)
    features = deepcopy(X[0])
    X = X[1:]
    return features, X


def the_kind_of_feature(feature):
    """

    :param feature: should be get from features[i]
    :return: 3 kinds of features.
    """
    number_kind = ['budget', 'id', 'popularity',
                   'revenue', 'release_date', 'runtime',
                   'vote_average', 'vote_count']
    vector_kind = ['genres', 'keywords', 'original_language',
                   'production_companies', 'production_countries',
                   'spoken_languages', 'status']
    if feature in number_kind:
        return 'number'
    elif feature in vector_kind:
        return 'vector'
    else:
        return 'string'


def fix(features, X):
    """

    :param X:
    :return: fixed X
    """
    fixed_X = deepcopy(X)
    for i in range(len(fixed_X)):
        if not X[i]:
            continue
        for j in range(len(fixed_X[i])):
            if the_kind_of_feature(features[j]) == 'string':
                fixed_X[i][j] = [0]
                # fixed_X[i][j] = np.array(fixed_X[i][j])

    # fix json vector features
    json_vec = []
    jsons = []
    json_vec.append(features.index('genres'))
    json_vec.append(features.index('keywords'))
    json_vec.append(features.index('production_companies'))
    json_vec.append(features.index('production_countries'))
    json_vec.append(features.index('spoken_languages'))
    for i in range(len(json_vec)):
        jsons.append(features[json_vec[i]])
    for json in json_vec:
        all_genres = []
        for i in range(len(X)):
            genre = eval(X[i][json])
            if not genre:
                continue
            for item in genre:
                # if not item['name'] in all_genres:
                all_genres.append(item['name'])
        List_set = set(all_genres)
        c = []
        for item in List_set:
            if all_genres.count(item) > 100:
                c.append(item)
        if not c:
            c = [0]
        all_genres = c
        Xs_genre = []
        for i in range(len(X)):
            x_genre = [0] * len(all_genres)
            genre = eval(X[i][json])
            if not genre:
                Xs_genre.append(x_genre)
                continue
            for item in genre:
                if item['name'] in all_genres:
                    index = all_genres.index(item['name'])
                    x_genre[index] = 1
            Xs_genre.append(x_genre)

        for i in range(len(fixed_X)):
            fixed_X[i][json] = Xs_genre[i]
            # fixed_X[i][json] = np.array(fixed_X[i][json])

    # fix other vector features
    other_vec = []
    for feature in features:
        if the_kind_of_feature(feature) == 'vector':
            if feature not in jsons:
                other_vec.append(features.index(feature))

    for vec in other_vec:
        all_vec = []
        for i in range(len(X)):
            if not X[i][vec]:
                continue
            if X[i][vec] not in all_vec:
                all_vec.append(X[i][vec])

        Xs_vec = []
        for i in range(len(X)):
            x_vec = [0] * len(all_vec)
            if not X[i][vec]:
                Xs_vec.append(x_vec)
                continue
            if X[i][vec] in all_vec:
                index = all_vec.index(X[i][vec])
                x_vec[index] = 1
            Xs_vec.append(x_vec)

        for i in range(len(fixed_X)):
            fixed_X[i][vec] = Xs_vec[i]
            # fixed_X[i][vec] = np.array(fixed_X[i][vec])

    # fix number vector features
    num_vecs = [feature for feature in features if (the_kind_of_feature(feature) == 'number')]
    num_vecs_index = []
    for i in range(len(num_vecs)):
        num_vecs_index.append(features.index(num_vecs[i]))

    #   fix release date
    index_date = features.index('release_date')
    for x in fixed_X:
        if x[index_date]:
            date = x[index_date].split('-')
            year, month, day = int(date[0]), int(date[1]), int(date[2])
            absdate = 365 * year + 30 * month + day
            x[index_date] = absdate

    for i in num_vecs_index:
        for x in fixed_X:
            if x[i]:
                x[i] = float(x[i])
            else:
                x[i] = 0

    for i in num_vecs_index:
        numbers = []
        for x in fixed_X:
            f = deepcopy(x[i])
            numbers.append(float(f))
        left, right = min(numbers), max(numbers)
        for x in fixed_X:
            f = deepcopy(x[i])
            x[i] = [(float(f) - left) / (right - left)]
            # x[i] = np.array(x[i])
    return fixed_X


def count_len(feature):
    """

    :param feature:
    :return:
    """
    count = 0
    already = []
    for i in range(len(feature)):
        if feature[i] not in already:
            count += 1
    vector_len = count
    return vector_len


def get_top_250():
    path = './movies.csv'
    movies = pd.read_csv(path)
    movies['total_votes'] = movies['vote_average'] * movies['vote_count']
    movies.sort_values('total_votes', ascending=False, inplace=True)
    Top250 = movies.head(250)
    Top250.to_csv('./movies_250.csv')


def pca_2d(K, dim):

    newpath = './movies_250.csv'
    # global features
    features, X = read_file(newpath)
    fixed_X = fix(features,X)
    nokeywords = deepcopy(fixed_X)
    for x in nokeywords:
        for i in range(21):
            if i in [0,2,4,5,6,10,11,15,16]:
                x[i] = [0]
    # print(nokeywords[0],'\n', nokeywords[1])
    k = K
    x_pca = deepcopy(nokeywords)
    for n in range(len(x_pca)):
        for f in range(len(x_pca[n])):
            # if f in []
            x_pca[n][f] = sum([nokeywords[n][f][_] for _ in range(len(nokeywords[n][f]))])
    pca = PCA(n_components=dim)
    pca_fixed_x = pca.fit_transform(x_pca)
    for x in pca_fixed_x:
        for f in x:
            f = [f]
    assignn = assign_250(K)
    diction = {}
    for i in range(K):
        diction[i] = []
    for i in range(len(pca_fixed_x)):
        a = pca_fixed_x[i]
        diction[assignn[i]].append(a)
    for i in range(K):
        plt.scatter(np.array(diction[i])[:, 0], np.array(diction[i])[:, 1])  # label='%d cluster' % i)
    # plt.legend()
    plt.show()


def assign_250(K):
    path = './movies.csv'
    movies = pd.read_csv(path)
    movies['total_votes'] = movies['vote_average'] * movies['vote_count']
    movies.sort_values('total_votes', ascending=False, inplace=True)
    Top250 = movies.head(250)
    Top250.to_csv('./movies_250.csv')
    newpath = './movies_250.csv'
    # global features
    features, X = read_file(newpath)
    fixed_X = fix(features,X)

    k = K
    errs = []

    U = random.sample(fixed_X, k)
    alloc, er = kmeans.iteration(fixed_X, U)

    assignments = [[i] for i in range(len(fixed_X))]
    for i in range(len(fixed_X)):
        for j in range(k):
            if fixed_X[i] in alloc[j]:
                assignments[i].append(j)
    name = ['id', 'label']
    test = pd.DataFrame(columns=name, data=assignments)
    test.to_csv('./output.csv')
    assign = np.array(assignments)
    assignn = assign[:,1]
    return assignn


def compare_kmeans_kmeanspp(K):
    path = './movies.csv'
    features, X = read_file(path)
    fixed_x = fix(features, X)
    k = K
    errs = []
    errspp = []
    # kmeans
    for i in range(k):
        print("k = ", i+1)
        U = random.sample(fixed_x, i+1)
        alloc, er = kmeans.iteration(fixed_x, U)
        errs.append(er)
    kk = np.linspace(1, k, k)
    plt.plot(kk, errs, label='k-means')
    # kmeans++
    for i in range(k):
        print("k = ", i+1)
        U = [0]*(i+1)
        U[0] = random.sample(fixed_x, 1)[0]
        U = kmeans.get_k_means_pp(fixed_x, U)
        alloc, er = kmeans.iteration(fixed_x, U)
        errspp.append(er)
    kk = np.linspace(1, k, k)
    plt.plot(kk, errspp, label='k-means++')
    plt.legend()
    plt.show()


def find_k(K):

    f = pca_to_1d()
    fixed_x = [0]*len(f)
    for i in range(len(f)):
        fixed_x[i] = [f[i][0]]
    k = K
    errs = []
    errspp = []
    run_k('./movies_250.csv', K)
    for i in range(k):
        print("k = ", i+1)
        U = random.sample(fixed_x, i+1)
        alloc, er = kmeans.iteration(fixed_x, U)
        errs.append(er)
    return errs


def run_k_means_pp(K):
    path = './movies.csv'
    features, X = read_file(path)
    fixed_x = fix(features, X)
    k = K
    # initialize U[0
    U = [0] * (k)
    U[0] = random.sample(fixed_x, 1)[0]
    # kmeans++
    U = kmeans.get_k_means_pp(fixed_x, U)
    alloc, er = kmeans.iteration(fixed_x, U)

    assignments = [[i] for i in range(len(fixed_x))]
    for i in range(len(fixed_x)):
        for j in range(k):
            if fixed_x[i] in alloc[j]:
                assignments[i].append(j)
    name = ['id', 'label']
    test = pd.DataFrame(columns=name, data=assignments)
    test.to_csv('./output_k++.csv')


def run_k(path,K):
    features, X = read_file(path)
    fixed_x = fix(features, X)
    k = K
    errs = []

    U = random.sample(fixed_x, k)
    alloc, er = kmeans.iteration(fixed_x, U)

    assignments = [[i] for i in range(len(fixed_x))]
    for i in range(len(fixed_x)):
        for j in range(k):
            if fixed_x[i] in alloc[j]:
                assignments[i].append(j)
    name = ['id', 'label']
    test = pd.DataFrame(columns=name, data=assignments)
    test.to_csv('./output_1d.csv')


def run_k_means(K):
    path = './movies.csv'
    features, X = read_file(path)
    fixed_x = fix(features, X)
    k = K
    errs = []

    U = random.sample(fixed_x, k)
    alloc, er = kmeans.iteration(fixed_x, U)

    assignments = [[i] for i in range(len(fixed_x))]
    for i in range(len(fixed_x)):
        for j in range(k):
            if fixed_x[i] in alloc[j]:
                assignments[i].append(j)
    name = ['id', 'label']
    test = pd.DataFrame(columns=name, data=assignments)
    test.to_csv('./output.csv')


def pca_to_1d():
    path = './movies_250.csv'
    features, X = read_file(path)
    fixed_X = fix(features, X)
    nokeywords = deepcopy(fixed_X)
    for x in nokeywords:
        for i in range(21):
            if i in [0,2,4,5,6,10,11,15,16]:
                x[i] = [0]
    # k = K
    x_pca = deepcopy(nokeywords)
    for n in range(len(x_pca)):
        for f in range(len(x_pca[n])):
            # if f in []
            x_pca[n][f] = sum([nokeywords[n][f][_] for _ in range(len(nokeywords[n][f]))])
    pca = PCA(n_components=1)
    pca_fixed_x = pca.fit_transform(x_pca)
    return pca_fixed_x


def run_k_means_pp_disagreement(K):
    path = './movies_250.csv'
    features, X = read_file(path)
    fixed_x = fix(features, X)
    k = K
    # initialize U[0
    U = [0] * (k)
    U[0] = random.sample(fixed_x, 1)[0]
    # kmeans++
    U = kmeans.get_k_means_pp(fixed_x, U)
    alloc, er = kmeans.iteration(fixed_x, U)

    assignments = [[i] for i in range(len(fixed_x))]
    for i in range(len(fixed_x)):
        for j in range(k):
            if fixed_x[i] in alloc[j]:
                assignments[i].append(j)
    name = ['id', 'label']
    test = pd.DataFrame(columns=name, data=assignments)
    test.to_csv('./output_k++_disagreement.csv')
    ppath = './output_k++_disagreement.csv'
    movies = pd.read_csv(ppath)


if __name__ == '__main__':
    run_k_means_pp(10)
