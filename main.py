from sklearn.linear_model import LogisticRegression
from random import random
from math import log
import numpy as np
BC_SCALED = 'data/breast-cancer_scale.txt'


def _read_data(all_data):
    y, x = [], []
    with open(all_data) as f:
        for line in f:
            row = line.strip().split(' ')
            y.append(int(row[0]))
            x.append([float(f[f.find(':') + 1:]) for f in row[1:]])
    return y, x


def extract_binary_lr_model(m, d):
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
    labels, features = _read_data(BC_SCALED)
    clf = LogisticRegression(random_state=0, solver='lbfgs').fit(features, labels)
    # this the probability of second label in Y, and the probability ot first label is obtained by
    # 1 - p(second label)
    print(f'model coefficients are: {clf.coef_}, beta = {clf.intercept_}')  # this is the w vector
    # noinspection PyTypeChecker
    print(f'the extracted coeficients are {extract_binary_lr_model(clf, 10)}')
