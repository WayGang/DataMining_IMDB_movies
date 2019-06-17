# Copyright2019 Gang Wei wg0502@bu.edu

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
import sys
import csv
import oned


if __name__ == '__main__':
      print("------Processing source.py------")

      path = sys.argv[1]
      input_k = int(sys.argv[2])
      init = sys.argv[3]

      print("path =", path, "\n",
            "k =", input_k, "\n",
            "init =", init)

      if path == './movies_250.csv':
            if init == 'gettop250':
                  data_prepocess.get_top_250()
            if init == 'random':
                  data_prepocess.assign_250(input_k)
            if init == 'plot2d':
                  data_prepocess.pca_2d(input_k,2)

      elif path == './movies.csv':
            if init == '1d':
                  if input_k < 4:
                        oned.run(input_k)
                  else:
                        oned.find_k_for_1d(input_k)

            elif init == 'random':
                  data_prepocess.run_k_means(input_k)
            elif init == 'k-means++':
                  data_prepocess.run_k_means(input_k)
            elif init == 'compare':
                  data_prepocess.compare_kmeans_kmeanspp(input_k)
            elif init == 'disagreement':
                  data_prepocess.run_k_means_pp_disagreement(input_k)
                  ppath = './output_k++_disagreement.csv'
                  moviesa = pd.read_csv(ppath)
                  a = moviesa['label'].tolist()

                  opath = './output_1d.csv'
                  moviesb = pd.read_csv(opath)
                  b = moviesb['label'].tolist()

                  dictp = {}
                  dicto = {}
                  for i in range(6):
                        dictp[i] = []
                        dicto[i] = []

                  for i in range(len(a)):
                        for j in range(6):
                              if a[i] == j:
                                    dictp[j].append(i)
                              if b[i] == j:
                                    dicto[j].append(i)

                  pp = sorted(dictp.items(), key=lambda l: len(l[1]))
                  oo = sorted(dicto.items(), key=lambda l: len(l[1]))
                  I = 0
                  for i in range(len(pp)):
                        for j in range(len(pp[i][1])):
                              if pp[i][1][j] not in oo[i][1]:
                                    I += 1
                  for i in range(len(oo)):
                        for j in range(len(oo[i][1])):
                              if oo[i][1][j] not in pp[i][1]:
                                    I += 1
                  print("The disagreement distance is:", I,
                        "of 250 movies using",input_k,"means clustering")
