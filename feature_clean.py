
# import my libraries
import sys
sys.path.append("../..")
from util import log


# import libraries
import pandas as pd



# create dummies for categorical variables
def create_dummies(col_list, df):
    
    df_extend = df.copy()
    
    for col in col_list:
        dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True)
        df_extend = pd.concat([df_extend, dummies], axis=1, sort=True).drop(labels=col, axis=1)
    
    return df_extend



def reconcile_splits(X_t, X_v):
    
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



# get a baseline of cleaned data
def baseline(df, assign):
    
#     # no nulls
#     cat_nulls = check_nulls(cat, df)
#     quant_nulls = check_nulls(quant, df)
    
#     categorical_nulls(cat_nulls, df)
#     quantitative_nulls(quant_nulls, df)

    cat = [item[0] for item in assign.col_map if item[1] == 'cat']
    
    df = create_dummies(cat, df)
    
    return df



# get baseline of cleaned data train / val / test
def baseline_train_val_test(assign):
    
    X_t = baseline(pd.read_csv("data/X_train.csv"), assign)
    X_v = baseline(pd.read_csv("data/X_val.csv"), assign)
    X_te = baseline(pd.read_csv("data/X_test.csv"), assign)
    
    X_v, v_drop = reconcile_splits(X_t, X_v)
    X_te, te_drop = reconcile_splits(X_t, X_te)

    X_t.to_csv("data/X_train.csv", index=False)
    X_v.to_csv("data/X_val.csv", index=False)
    X_te.to_csv("data/X_test.csv", index=False)
    
    log("Dropped from validation set [column, count]: " + str(v_drop), __name__, "info")
    log("Dropped from testing set [column, count]: " + str(te_drop), __name__, "info")




