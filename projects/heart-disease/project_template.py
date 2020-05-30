
# This file holds all of the constant values to configure the project setup

# FIT_TYPE_REGRESSION = "reg"
# FIT_TYPE_BINARY_CLASSIFICATION = "bclass"
# FIT_TYPE_CLASSIFICATION = "class"

# EVAL_R2 = "r2"
# EVAL_LOGLOSS = "logloss"
# EVAL_AUC = "auc"

# ------------------------

# import my libraries
import sys
sys.path.append("../..")
from util import log
from train_test_validate import train_val_test_split
import feature_check as fc
import feature_clean as fclean
from assign import Assign
from train import train, train_bayesian_opt



FIT_TYPE = "binaryclass"   # "regression", "multiclass"
EVAL = "logloss"   #  "accuracy", "r2", "auc"
PATH = "data/heart.csv"
TARGET = "target"

X_TRAIN_PATH = "data/X_train.csv"



# pipeline start
log("\n\n\nPipeline start", __name__, "info")



# train / test / val split
train_val_test_split(PATH, TARGET, 0.2, 0.15)

# instantiate Assign object to track column assignments
x_train_assign = Assign(X_TRAIN_PATH)
x_train_assign.log()

# summarize the data
fc.summary(X_TRAIN_PATH, x_train_assign)

# correct assignment of columns
map_columns = {'sex': 'cat', 'cp': 'cat', 'fbs': 'cat', 'restecg': 'cat', 'exang': 'cat',
                'slope': 'cat', 'ca': 'cat', 'thal': 'cat'}

x_train_assign.remap_force(map_columns)
x_train_assign.log()

# summarize the data
fc.summary(X_TRAIN_PATH, x_train_assign)

# baseline clean data
fclean.baseline_train_val_test(x_train_assign)

x_train_assign.remap()
x_train_assign.log()
fc.summary(X_TRAIN_PATH, x_train_assign)


# instantiate Assign object to track column assignments
# x_train_assign = Assign(X_TRAIN_PATH)
# x_train_assign.log()

# train()

train_bayesian_opt()



# pipeline complete
log("Pipeline complete\n\n\n", __name__, "info")



