# import libraries
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split


# XXX parameterize path
# import data
data = pd.read_csv("heart.csv")





# train validate test split
def train_val_test_split(df, target, val, test):
    
    X = df.drop([target], axis=1)
    y = df[target]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=val+test, random_state=42)
    
    val_test_split = test / (val + test)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_test_split, random_state=42)
    
    return [X_train, y_train, X_val, y_val, X_test, y_test]

    #y_val = y_val.to_numpy()






X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(data, 'target', 0.2, 0.15)





# count null values
def count_nulls(df):
    return [df[col].isna().sum() for col in df.columns]


# assign quantitative and categorical columns by dtype
def assign_col(df):
    return ['cat' if df[col].dtype == object else 'quant' if df[col].dtype in [int, float]
            else '???' for col in df.columns]


# return data types
def data_types(df):
    return [df[col].dtype for col in df.columns]


# return summary statistics
def summary_stats(df):
    
    stats_list = []
    
    for col in df.columns:
        
        stats = ""
        
        if df[col].dtype in [int, float]:
            
            stats += "min: " + str(df[col].min())
            stats += ", max: " + str(df[col].max())
            stats += ", mean: " + str(round(df[col].mean(), 4))
            
        elif df[col].dtype == object:
            
            count = df[col].value_counts(sort = True)
            percent = df[col].value_counts(normalize = True, sort = True)

            values = pd.DataFrame({'count': count, 'percent': percent})

            cat_count = count.shape[0]

            for i, r in values.iterrows():
                stats += "{0}: {1} {2}%,   ".format(i, r['count'], round(r['percent'] * 100, 2))
        
        else:
            stats = "not correctly processed based on data type"
            
        stats_list.append(stats[:-4])
    
    return stats_list


# assemble all pieces together
def summary(df):
    
    n, m = df.shape[0], df.shape[1]
    
    print("n = {0}, m = {1}".format(n, m))
    
    names = df.columns
    nulls = count_nulls(df)
    assign = assign_col(df)
    dtypes = data_types(df)
    statistics = summary_stats(df)
    
    frame = {'Column Name': names, 'Null Count': nulls, 'Data Assign': assign,
             'Data Type': dtypes, 'Summary Stats': statistics}
    
    print(tabulate(pd.DataFrame(frame), headers='keys', tablefmt='psql'))





data.info()





summary(data)





summary(X_train)





# check for null values
def check_nulls(col_list, df):
    return [col for col in col_list if df[col].isna().sum() > 1]


# fill null values
def categorical_nulls(col_list, df):
    for col in col_list:
        df[col] = df[col].fillna('NA')

def quantitative_nulls(col_list, df):
    for col in col_list:
        df[col] = df[col].fillna(df[col].mean())


# 
# def baseline(df):
    
#     cat = categorical_columns(df)
#     quant = quantitative_columns(df)
    
#     # no nulls
#     cat_nulls = check_nulls(cat, df)
#     quant_nulls = check_nulls(quant, df)
    
#     categorical_nulls(cat_nulls, df)
#     quantitative_nulls(quant_nulls, df)
    
#     df = create_dummies(cat, df)
    
#     return df.iloc[:, 1:]





# remove item from a set without throwing an error
# if it does not exist
def remove_safe(lst, value):
    try:
        lst.remove(value)
    except KeyError:
        pass
    
    return lst


# identify quantitative and categorical columns by dtype
def detect_col_type(df):
    cat = {col for col in df if df[col].dtype == object}
    quant = {col for col in df if df[col].dtype in [int, float]}
    
    return [cat, quant]


# create dummies for categorical variables
def create_dummies(col_list, df):
    
    df_extend = df.copy()
    
    # if the column contains values outside of 0 and 1
    #col_list = [col for col in col_list if df[col] not in [0, 1]]
    
    for col in col_list:
        dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True)
        df_extend = pd.concat([df_extend, dummies], axis=1, sort=True).drop(labels=col, axis=1)
    
    return df_extend


def ReconcileSplits(X_t, X_v):
    
    in_train_not_val = [col for col in X_t.columns if col not in X_v.columns]
    in_val_not_train = [col for col in X_v.columns if col not in X_t.columns]
    
    new_cats = []
    
    for col in in_train_not_val:
        X_v[col] = 0
    
    for col in in_val_not_train:
        new_cats.append([col, X_v[col].sum()])
        X_v = X_v.drop([col], axis=1)
    
    # ensure the same column order
    X_v = X_v[X_t.columns]
    
    return [X_v, new_cats]

#     return [in_train_not_val, in_val_not_train]


# 
def baseline(df, assign = None):
    
    cat, quant = detect_col_type(df)
    
    if type(assign) == dict:
        
        for item in assign.items():
            if item[1] == 'cat':
                quant = remove_safe(quant, item[0])
                cat.add(item[0])
            elif item[1] == 'quant':
                cat = remove_safe(cat, item[0])
                quant.add(item[0])
            else:
                raise Exception("column must be assigned to 'cat' or 'quant'")
    
#     # no nulls
#     cat_nulls = check_nulls(cat, df)
#     quant_nulls = check_nulls(quant, df)
    
#     categorical_nulls(cat_nulls, df)
#     quantitative_nulls(quant_nulls, df)
    
    df = create_dummies(cat, df)
    
    return df #df.iloc[:, 1:]


# manage splits through transformations
def baseline_train_val_test(X_t, X_v, X_te, assign):
    
    X_t = baseline(X_t, assign)
    X_v = baseline(X_v, assign)
    X_te = baseline(X_te, assign)
    
    X_v, v_drop = ReconcileSplits(X_t, X_v)
    X_te, te_drop = ReconcileSplits(X_t, X_te)
    
    
    return [X_t, X_v, X_te, v_drop, te_drop]






map_columns = {'sex': 'cat', 'cp': 'cat', 'fbs': 'cat', 'restecg': 'cat', 'exang': 'cat',
                'slope': 'cat', 'ca': 'cat', 'thal': 'cat'}

X_t, X_v, X_te, v_drop, te_drop = baseline_train_val_test(X_train, X_val, X_test, map_columns)






from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
#from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier ###
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier    
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
#from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
#from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
#from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
#from sklearn.naive_bayes import MultinomialNB  
#from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
### from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC

# import issue
# from sklearn.mixture import DPGMM
# from sklearn.mixture import GMM 
# from sklearn.mixture import GaussianMixture
# from sklearn.mixture import VBGMM


# from sklearn.multiclass import OutputCodeClassifier
# from sklearn.multiclass import OneVsOneClassifier
# from sklearn.multiclass import OneVsRestClassifier





from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, log_loss





classifiers = [['kNN', {'n_neighbors': [3,5,7,9], 'weights': ['uniform', 'distance'],
                                                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'p': [1,2,3]}]]






for classifier in classifiers:
    
    for k in classifier[1]['n_neighbors']:
        for weights in classifier[1]['weights']:
            for algorithm in classifier[1]['algorithm']:
                for p in classifier[1]['p']:
                    
                    model = KNeighborsClassifier(n_neighbors=k, weights=weights, algorithm=algorithm, p=p)
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_val)
                    
                    # parameterize what you want to evaluate / see
                    #print(classification_report(y_val, y_pred))
                    print('Parameters: k={}, weights={}, algorithm={}, p={}'.format(k, weights, algorithm, p))
                    print('Accuracy: {:.2%}, Log Loss: {:.4}'.format(accuracy_score(y_val, y_pred), log_loss(y_val, y_pred)))






Parameters: k=5, weights=uniform, algorithm=auto, p=1 Accuracy: 72.13%, Log Loss: 9.626

Parameters: k=5, weights=uniform, algorithm=ball_tree, p=1 Accuracy: 72.13%, Log Loss: 9.626

Parameters: k=5, weights=uniform, algorithm=kd_tree, p=1 Accuracy: 72.13%, Log Loss: 9.626

Parameters: k=5, weights=uniform, algorithm=brute, p=1 Accuracy: 72.13%, Log Loss: 9.626

Parameters: k=5, weights=distance, algorithm=auto, p=1 Accuracy: 72.13%, Log Loss: 9.626

Parameters: k=5, weights=distance, algorithm=ball_tree, p=1 Accuracy: 72.13%, Log Loss: 9.626

Parameters: k=5, weights=distance, algorithm=kd_tree, p=1 Accuracy: 72.13%, Log Loss: 9.626

Parameters: k=5, weights=distance, algorithm=brute, p=1 Accuracy: 72.13%, Log Loss: 9.626

Clear winners k = 5 p = 1





# def ScaleTest(X_t, y_t, X_v, y_v):
    
#     scales = [('Not Scaled/Normalized', np.nan), ('MaxAbsScaler', pp.MaxAbsScaler()), ('MinMaxScaler', pp.MinMaxScaler()),
#               ('L1 Normalizer', pp.Normalizer(norm='l1')), ('L2 Normalizer', pp.Normalizer(norm='l2')),
#               ('PowerTransformer', pp.PowerTransformer()), ('Uniform QuantileTransformer', pp.QuantileTransformer(output_distribution='uniform')),
#               ('Normal QuantileTransformer', pp.QuantileTransformer(output_distribution='normal')), ('RobustScaler', pp.RobustScaler()),
#               ('StandardScaler', pp.StandardScaler())]
    
#     for scaler in scales:
        
#         X_t_scale = X_t
#         X_v_scale = X_v
        
#         if scaler[0] != 'Not Scaled/Normalized':
#             X_t_scale = Scale(X_t, scaler[1])
#             X_v_scale = Scale(X_v, scaler[1])
        
#         reg = LinearRegression()
#         reg.fit(X_t_scale, y_t)

#         y_p = reg.predict(X_v_scale)
        
#         print('\n' + scaler[0])
#         LREvaluate(y_v, y_p, X_v_scale.shape[1])
        





import sys
print(sys.path)






