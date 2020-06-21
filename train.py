
# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s:   %(asctime)s\n%(message)s')
file_handler = logging.FileHandler('logs/train.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# import my libraries
import sys
sys.path.append("../..")
from eval_model import evaluate

# import libraries
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from GPyOpt.methods import BayesianOptimization



def loadData():

    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv").values.ravel()
    X_val = pd.read_csv("data/X_val.csv")
    y_val = pd.read_csv("data/y_val.csv").values.ravel()

    return X_train, y_train, X_val, y_val


def train(eval = True, params = None):

    X_train, y_train, X_val, y_val = loadData()

    if params == None:
        model = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform', algorithm = 'brute', p = 2)
    else:
        model = KNeighborsClassifier(n_neighbors = params['n_neighbors'], weights = 'uniform', algorithm = 'brute', p = params['p'])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:,1]   #binary classifier specific

    if eval == True:
        evaluate("binaryclass", y_val, y_pred, y_prob)



# http://krasserm.github.io/2018/03/21/bayesian-optimization/
# scoring: https://scikit-learn.org/stable/modules/model_evaluation.html
def train_bayesian_opt(eval = True):

    X_train, y_train, X_val, y_val = loadData()

    knn_params = [{'name': 'n_neighbors', 'type': 'discrete', 'domain': (3,5,7,9,11,13)},
        {'name': 'p', 'type': 'discrete', 'domain': (1,2,3)}]

    # Optimization objective 
    def cv_score(parameters):
        parameters = parameters[0]
        score = cross_val_score(
                    KNeighborsClassifier(n_neighbors=int(parameters[0]),
                                        weights = 'uniform',
                                        algorithm = 'brute',
                                        p=int(parameters[1])), 
                    X_train, y_train, scoring='accuracy').mean()
        score = np.array(score)
        return score

    optimizer = BayesianOptimization(f=cv_score, 
                                     domain=knn_params,
                                     model_type='GP',
                                     acquisition_type ='EI',
                                     acquisition_jitter = 0.001,
                                     maximize=True)

    optimizer.run_optimization(max_iter=20)

    logger.info(str(optimizer.Y))
    logger.info(str(optimizer.X))
    logger.info(str(optimizer.X[-1]))

    if eval == True:
        train(params = {'n_neighbors': int(optimizer.X[-1][0]), 'p': int(optimizer.X[-1][1])})


