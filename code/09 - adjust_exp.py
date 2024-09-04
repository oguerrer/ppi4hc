import os, warnings
import pandas as pd
import numpy as np
from scipy.signal import detrend

warnings.simplefilter("ignore")

home =  os.getcwd()[0:-4]


###########################################################################
###########################################################################
#
# Adjust expenditure data by deflating, normalizing, and detrending
#
###########################################################################
###########################################################################


# We are going to adjust expenditure by computing per capita values,
# deflate the time series and finally detrend them (linear trend)


df = pd.read_csv(home + '/data/clean/Variables/add_variables.csv', encoding='utf-8-sig')

cols = df.columns
colYears = [col for col in df.columns if col.isnumeric()]
years_indices = df.columns.isin(colYears)

new_rows = []

# use as base year the mid-point of the time series

base = df.loc[df.Variable=="CPI","2019"].values[0]

for i in colYears :
    df.loc[df.Variable=="CPI",i] = df.loc[df.Variable=="CPI",i].values[0]/base

# deflate and adjust for population growth

df_exp = pd.read_csv(home + "/data/clean/Expenditure/coneval_exp_pivot.csv", encoding='utf-8-sig')

df_eje = df_exp.loc[df_exp.Presupuesto=="Ejercido",:].copy().reset_index(drop=True)

dff = df_eje.copy()

for y in colYears:
    cpi =  df.loc[df.Variable=="CPI",y]
    pop = df.loc[df.Variable=="POP",y]
    dff[y] = df_eje[y].values / pop.values / cpi.values

dff.to_csv(home + "/data/clean/Expenditure/coneval_exp_def_pc.csv", encoding='utf-8-sig', index=False)

# Code to detrend
# !! negative value for 12-S039 in 2016

df_final = dff.copy()
df_final[colYears] = np.clip([detrend(serie)+np.mean(serie) for serie in dff[colYears].values], a_min=0, a_max=None)

# Set millions of pesos

df_final[colYears] = df_final[colYears].values * 1000000

df_final.to_csv(home + "/data/clean/Expenditure/coneval_exp_detrend.csv", encoding='utf-8-sig', index=False)
