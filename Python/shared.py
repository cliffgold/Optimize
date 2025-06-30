import numpy as np
import pandas as pd

nrgs  = np.array(['Solar', 'Wind', 'Nuclear', 'Gas', 'Coal', 'Battery'])

# used by debug, and other places. *.csv converts 1.0 to 1, so we need to convert it back. 
def df_int_to_float(df):
# Identify integer columns and convert them to float
    int_cols = df.select_dtypes(include=['int']).columns
    df[int_cols] = df[int_cols].astype(float)
    return df
