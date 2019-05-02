from sklearn.linear_model import LogisticRegression
from random import random
from math import log
import numpy as np


def _bc_data():
    y, x = [], []
    with open('data/breast-cancer_scale.txt') as f:
        for line in f:
            row = line.strip().split(' ')
            y.append(int(row[0]))
            x.append([float(f[f.find(':') + 1:]) for f in row[1:]])
    return y, x


def extract_binary_lr(m, d):
    """
    :param m: the model to extract
    :param d: dimension size of features
    :return: the extract model parameters
    """
    x = [[random() for _ in range(d)] for _ in range(d + 1)]  # create d+1 feature vectors, each with d features
    probs = [p[1] for p in m.predict_proba(x)]
    for f in x:
        f.append(1)
    return np.linalg.solve(x, list(map(lambda c: -log((1 - c) / c), probs)))  # sigma-1, defined in Tramèr 4.1.1


if __name__ == '__main__':
    labels, features = _bc_data()
    clf = LogisticRegression(random_state=0, solver='lbfgs').fit(features, labels)
    # this the probability of second label in Y, and the probability ot first label is obtained by 1 - p(second label)
    print(f'model coefficients are: {clf.coef_}, beta = {clf.intercept_}')  # this is the w vector
    print(f'the extracted coeficients are {extract_binary_lr(clf, 10)}')
