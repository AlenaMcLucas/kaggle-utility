
# Test different features scaling methods
# Need another function to select the method outright

# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s:   %(asctime)s\n%(message)s')
file_handler = logging.FileHandler('logs/features.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# import libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing as pp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from tabulate import tabulate



def scale(X_t, X_v, scaler, X_te = pd.DataFrame()):

    cols = X_t.columns
    
    scaler_fit = scaler.fit(X_t)
    X_t_s = scaler_fit.transform(X_t)
    X_v_s = scaler_fit.transform(X_v)
    
    if X_te.empty == True:
        return pd.DataFrame(X_t_s, columns=cols), pd.DataFrame(X_v_s, columns=cols)
    else:
        X_te_s = scaler_fit.transform(X_te)
        return pd.DataFrame(X_t_s, columns=cols), pd.DataFrame(X_v_s, columns=cols), pd.DataFrame(X_te_s, columns=cols),



# need to update to have alorithm work for non-sklearn models
def scale_test(model = None, scales = None, rank = "accuracy", save = False):

    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv")
    X_val = pd.read_csv("data/X_val.csv")
    y_val = pd.read_csv("data/y_val.csv")

    if save == False and model != None:

        scale_tracker = []
        
        if scales == None:
            scales = [('Not Scaled/Normalized', np.nan), ('MaxAbsScaler', pp.MaxAbsScaler()), ('MinMaxScaler', pp.MinMaxScaler()),
                      ('L1 Normalizer', pp.Normalizer(norm='l1')), ('L2 Normalizer', pp.Normalizer(norm='l2')),
                      ('PowerTransformer', pp.PowerTransformer()), ('Uniform QuantileTransformer', pp.QuantileTransformer(output_distribution='uniform')),
                      ('Normal QuantileTransformer', pp.QuantileTransformer(output_distribution='normal')), ('RobustScaler', pp.RobustScaler()),
                      ('StandardScaler', pp.StandardScaler())]
        

        for scaler in scales:
            
            X_t_scale = X_train
            X_v_scale = X_val
            
            if scaler[0] != 'Not Scaled/Normalized':
                X_t_scale, X_v_scale = scale(X_train, X_val, scaler[1])
            
            #model = algorithm
            model.fit(X_t_scale, y_train)
            y_pred = model.predict(X_v_scale)

            # score predictions
            score = np.nan
            if rank == "accuracy":
                score = accuracy_score(y_val, y_pred)
            elif rank == "logloss":
                score = log_loss(y_val, y_pred)
            
            scale_tracker.append((scaler[0], score))


        scale_tracker.sort(key = lambda x: x[1], reverse = (True if rank == "accuracy" else False ) )
        logger.info(tabulate(scale_tracker, tablefmt="psql"))

    # if user wants the scaler to save over the data and there is only one scaler selected, then do so
    if save == True and len(scales) == 1:

        X_test = pd.read_csv("data/X_test.csv")
        
        X_t_scale, X_v_scale, X_te_scale = scale(X_train, X_val, scales[0][1], X_test)

        X_t_scale.to_csv("data/X_train.csv", index=False)
        X_v_scale.to_csv("data/X_val.csv", index=False)
        X_te_scale.to_csv("data/X_test.csv", index=False)

        logger.info("Saved scaled data")





