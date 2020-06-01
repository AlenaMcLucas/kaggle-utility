
# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s:   %(asctime)s\n%(message)s')
file_handler = logging.FileHandler('logs/train.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# import libraries
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from GPyOpt.methods import BayesianOptimization



def train():

    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv")
    X_val = pd.read_csv("data/X_val.csv")
    y_val = pd.read_csv("data/y_val.csv")

    model = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform', algorithm = 'brute', p = 2)
                                # leaf_size = 30, metric = 'minkowski'

    model.fit(X_train, y_train.values.ravel())

    y_pred = model.predict(X_val)

    logger.info(str(accuracy_score(y_val.values.ravel(), y_pred)))



# http://krasserm.github.io/2018/03/21/bayesian-optimization/
# scoring: https://scikit-learn.org/stable/modules/model_evaluation.html
def train_bayesian_opt():

    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv")
    X_val = pd.read_csv("data/X_val.csv")
    y_val = pd.read_csv("data/y_val.csv")

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
                    X_train, y_train.values.ravel(), scoring='accuracy').mean()
        score = np.array(score)
        return score

    optimizer = BayesianOptimization(f=cv_score, 
                                     domain=knn_params,
                                     model_type='GP',
                                     acquisition_type ='EI',
                                     acquisition_jitter = 0.05,
                                     maximize=True)

    optimizer.run_optimization(max_iter=20)

    logger.info(str(optimizer.Y))
    logger.info(str(optimizer.X))

