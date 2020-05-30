
# import my libraries
import sys
sys.path.append("../..")
from util import log


# import libraries
import pandas as pd
from tabulate import tabulate



# count null values
def count_nulls(df):
    return [df[col].isna().sum() for col in df.columns]



# return data types
def data_types(df):
    return [df[col].dtype for col in df.columns]



# return summary statistics
def summary_stats(df, assign):
    
    stats_list = []
    
    for assi in assign.col_map:
        
        stats = ""
        
        if assi[1] == "quant":

            stats = "min: {}, max: {}, mean: {:f}".format(df[assi[0]].min(), df[assi[0]].max(), df[assi[0]].mean())
            
        elif assi[1] == "cat":
            
            count = df[assi[0]].value_counts(sort = True) # .astype('int64')
            percent = df[assi[0]].value_counts(normalize = True, sort = True)

            values = pd.DataFrame({"count": count, "percent": percent})

            cat_count = count.shape[0]

            for i, r in values.iterrows():
                stats += "{}: {:.0f} {:.2%},   ".format(i, r['count'], r['percent'])
        
        else:

            stats = "not correctly assigned"
            
        stats_list.append(stats[:-4])
    
    return stats_list



# assemble all pieces together
def summary(path, assignment):

    df = pd.read_csv(path)
    
    n, m = df.shape[0], df.shape[1]
    
    log("n = {0}, m = {1}".format(n, m), __name__, "info")
    
    names = df.columns
    nulls = count_nulls(df)
    assign = [tup[1] for tup in assignment.col_map]
    dtypes = data_types(df)
    statistics = summary_stats(df, assignment)
    
    frame = {"Column Name": names, "Null Count": nulls, "Data Assign": assign,
             "Data Type": dtypes, "Summary Stats": statistics}
    
    log(tabulate(pd.DataFrame(frame), headers="keys", tablefmt="psql"), __name__, "info")



