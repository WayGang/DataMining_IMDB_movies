# Copyright2019 Gang Wei wg0502@bu.edu

import random
import numpy as np
import kmeans
import csv
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt

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


def fix(X):
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
                fixed_x[i][j] = np.array(fixed_X[i][j])

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
                if not item['name'] in all_genres:
                    all_genres.append(item['name'])

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
            fixed_x[i][json] = np.array(fixed_X[i][json])

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


if __name__ == '__main__':
    path = '/Users/gangwei/Desktop/565project1/movies.csv'
    features, X = read_file(path)

    fixed_x = fix(X)
    nokeywords = deepcopy(fixed_x)
    for x in nokeywords:
        x[4] = [0]
        # x[9] = [0]'''

    k = 10

    errs = []
    for i in range(k):
        U = random.sample(nokeywords, i+1)
        [alloc, er] = kmeans.iteration(nokeywords, U)
        errs.append(er)
        print(alloc)
    kk = np.linspace(1, k, k)
    plt.plot(kk, errs)
    plt.show()

    # print(kmeans.getCluster(fixed_x, U))
    # count_len(X[0])
    '''name = ['aaaa'] * 20
    test = pd.DataFrame(columns=name, data=nokeywords)
    test.to_csv('/Users/gangwei/Desktop/565project1/movies_fixed_nokey.csv')'''
