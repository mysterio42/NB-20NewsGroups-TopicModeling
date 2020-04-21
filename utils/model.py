import glob
import os
import random
import string
from operator import itemgetter

import joblib
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB

from utils.plot import plot_cm

WEIGHTS_DIR = 'weights/'


def latest_modified_weight():
    """
    returns latest trained weight
    :return: model weight trained the last time
    """
    weight_files = glob.glob(WEIGHTS_DIR + '*')
    latest = max(weight_files, key=os.path.getctime)
    return latest


def load_model(path):
    """

    :param path: weight path
    :return: load model based on the path
    """

    with open(path, 'rb') as f:
        return joblib.load(filename=f)


def dump_model(model):
    model_name = WEIGHTS_DIR + 'NaivBay-' + generate_model_name(5) + '.pkl'
    with open(model_name, 'wb') as f:
        joblib.dump(value=model, filename=f, compress=3)
        print(f'Model saved at {model_name}')


def generate_model_name(size=5):
    """

    :param size: name length
    :return: random lowercase and digits of length size
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def find_optimal_alpha(features, labels):
    alpha_range = np.linspace(1e-5, 1e-1)
    scores = []
    scoring = {'f1_weighted': make_scorer(metrics.f1_score, average='weighted')}
    for alpha in alpha_range:
        model = MultinomialNB(alpha=alpha)
        all_scores = cross_validate(model, features, labels, cv=10, scoring=scoring, return_train_score=False)
        score = all_scores['test_f1_weighted'].mean()
        scores.append(score)

    res = zip(alpha_range, scores)
    opt_alpha, opt_score = max(res, key=itemgetter(1))

    return alpha_range, scores, opt_alpha, opt_score


def train_model(opt_alpha, features_train, labels_train, features_test, labels_test):
    model = MultinomialNB(alpha=opt_alpha)
    model.fit(features_train, labels_train)

    preds = model.predict(features_test)

    cm = confusion_matrix(labels_test, preds)
    plot_cm(cm)

    print(accuracy_score(labels_test, preds))

    ans = input('Do you want to save the model weight? ')
    if ans in ('yes', '1'):
        dump_model(model)

    return model


def predict_model(model, embed_features, categories):
    category = model.predict(embed_features)[0]
    return categories[category]
