
# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s:   %(asctime)s\n%(message)s')
file_handler = logging.FileHandler('logs/pipeline.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split



# train / validate / test split
# parameters
# path = string to data file, target = y variable column name,
# val = validation as % of total, test = test as % of total
# returns
# nothing, outputs files to data folder

def train_val_test_split(path, target, val, test):

	df = pd.read_csv(path)

	X = df.drop([target], axis=1)
	y = df[target]

	X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=val+test, random_state=42)

	val_test_split = test / (val + test)

	X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_test_split, random_state=42)
	

	X_train.to_csv("data/X_train.csv", index=False)
	X_val.to_csv("data/X_val.csv", index=False)
	X_test.to_csv("data/X_test.csv", index=False)
	y_train.to_csv("data/y_train.csv", index=False)
	y_val.to_csv("data/y_val.csv", index=False)
	y_test.to_csv("data/y_test.csv", index=False)
	

	logger.info("Train / validate / test split complete")
