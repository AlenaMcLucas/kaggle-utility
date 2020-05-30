
# import libraries
import numpy as np
import pandas as pd


# XXX parameterize path
# import data
data = pd.read_csv("heart.csv")


# identify quantitative and categorical columns by dtype
def categorical_columns(df):
    return [col for col in df if df[col].dtype == object]

def quantitative_columns(df):
    return [col for col in df if df[col].dtype in [int, float]]


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


# create dummies for categorical variables
def create_dummies(col_list, df):
    
    df_extend = df.copy()
    
    for col in col_list:
        dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True)
        df_extend = pd.concat([df_extend, dummies], axis=1, sort=True).drop(labels=col, axis=1)
    
    return df_extend


# 
def baseline(df):
    
    cat = categorical_columns(df)
    quant = quantitative_columns(df)
    
    # no nulls
    cat_nulls = check_nulls(cat, df)
    quant_nulls = check_nulls(quant, df)
    
    categorical_nulls(cat_nulls, df)
    quantitative_nulls(quant_nulls, df)
    
    df = create_dummies(cat, df)
    
    return df.iloc[:, 1:]