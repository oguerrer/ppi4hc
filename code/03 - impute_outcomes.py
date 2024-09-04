import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
warnings.simplefilter("ignore")

home =  os.getcwd()[0:-4]


##################################################
##################################################
#
# Imputes missing values using Gaussian processes
#
##################################################
##################################################

# To impute: 11-E017, 11-E039, 11-S243 (No), 12-S039 (No), 50-E012 (No)

df = pd.read_csv(home + "/data/clean/Outcomes/coneval_perf.csv", encoding='utf-8-sig')

df_orig = df.copy()


colYears = [col for col in df.columns if str(col).isnumeric()]
years = np.array([int(col) for col in df.columns if str(col).isnumeric()])
years_indices = df.columns.isin(colYears)

new_rows = []

for index, row in df.iterrows():
    
    print('Imputing missing values (if any) for indicator', index)
    
    observations = np.where(~row[colYears].isnull())[0]
    missing_values = np.where(row[colYears].isnull())[0]
    new_row = row.values.copy()
        
        
    vals = row[colYears].values.copy()
    
    x = years[observations]
    y = vals[observations]
    X = x.reshape(-1, 1)
    
    kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
    gp.fit(X, y)
    
    x_pred = years.reshape(-1,1)
    y_pred, sigma = gp.predict(x_pred, return_std=True)
    
    vals[missing_values] = y_pred[missing_values]
    new_row[years_indices] = vals
    new_rows.append(new_row)



dff = pd.DataFrame(new_rows, columns=df.columns)

dff.to_csv(home + "/data/clean/Outcomes/coneval_perf_imputed.csv", encoding='utf-8-sig', index=False)
