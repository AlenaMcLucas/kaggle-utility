
# This file holds all of the constant values to configure the project setup

# remove some logs if they exist
import os
for p in ["logs/features.log", "logs/train.log"]:
    if os.path.exists(p):
        os.remove(p)

# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s:   %(asctime)s\n%(message)s')
file_handler = logging.FileHandler('logs/pipeline.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# import my libraries
import sys
sys.path.append("../..")
from train_test_validate import train_val_test_split
from assign import Assign
import feature_check as fc
import feature_clean as fclean
from scale import scale, scale_test
from train import train, train_bayesian_opt

# import libraries
from sklearn.neighbors import KNeighborsClassifier



FIT_TYPE = "binaryclass"   # "regression", "multiclass"
EVAL = "logloss"   #  "accuracy", "r2", "auc"
PATH = "data/heart.csv"
X_TRAIN_PATH = "data/X_train.csv"
TARGET = "target"



# pipeline start
logger.info("\n\n\nPipeline start")



# train / test / val split
train_val_test_split(PATH, TARGET, 0.2, 0.15)

# scaler
scale_test(KNeighborsClassifier(n_neighbors = 5, weights = 'uniform', algorithm = 'brute', p = 2))

# # instantiate Assign object to track column assignments
# x_train_assign = Assign(X_TRAIN_PATH)
# x_train_assign.log()

# # summarize the data
# fc.summary(X_TRAIN_PATH, x_train_assign)

# # correct assignment of columns
# map_columns = {'sex': 'cat', 'cp': 'cat', 'fbs': 'cat', 'restecg': 'cat', 'exang': 'cat',
#                 'slope': 'cat', 'ca': 'cat', 'thal': 'cat'}

# x_train_assign.remap_force(map_columns)
# x_train_assign.log()

# # summarize the data
# fc.summary(X_TRAIN_PATH, x_train_assign)

# # baseline clean data
# fclean.baseline_train_val_test(x_train_assign)

# x_train_assign.remap()
# x_train_assign.log()
# fc.summary(X_TRAIN_PATH, x_train_assign)

# # train()

# train_bayesian_opt()



# pipeline complete
logger.info("Pipeline complete\n\n\n")



