
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



def scale(X_t, X_v, scaler):

    cols = X_t.columns
    
    scaler_fit = scaler.fit(X_t)
    X_t_s = scaler_fit.transform(X_t)
    X_v_s = scaler_fit.transform(X_v)
    
    return pd.DataFrame(X_t_s, columns=cols), pd.DataFrame(X_v_s, columns=cols)



# need to update to have alorithm work for non-sklearn models
def scale_test(model):

    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv")
    X_val = pd.read_csv("data/X_val.csv")
    y_val = pd.read_csv("data/y_val.csv")
    
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
        
        logger.info('\n' + scaler[0])
        # need to include evaluation output
        # need to style output, but is dependent on y variable and evaluation modules
        logger.info(accuracy_score(y_val, y_pred))



