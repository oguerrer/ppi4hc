import os, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

home =  os.getcwd()[0:-4]

#############################
#############################
#
# Computes performance metric
#
#############################
#############################

# We are going to first define the performance metrics, then impute missing values
# finally adjust imputations.


df = pd.read_csv(home + '/data/clean/Outcomes/coneval_oc_pivot.csv', encoding='utf-8-sig')

cols = df.columns
colYears = [col for col in df.columns if col.isnumeric()]
years_indices = df.columns.isin(colYears)

new_rows = []

df_pot = df.loc[df.Outcome=="Población Potencial",:].copy().reset_index(drop=True)
df_at = df.loc[df.Outcome=="Población Atendida",:].copy().reset_index(drop=True)

dict_max_pot = {i: np.nan for i in np.unique(df_pot.ID_RMC)}

# Max potential population plus average absolute difference of potential population
# over the years (2016-2022): denominator

for index,row in df_pot.iterrows():
    vals = row[colYears].values.astype(float)
    diff = abs(vals[1::] - vals[0:-1])
    meandiff = np.nanmean(diff)
    dict_max_pot[row["ID_RMC"]] = np.nanmax(vals) + meandiff

# Performance: population attended/ denominator

for index,row in df_at.iterrows():
    vals_at = row[colYears].values.astype(float)
    nvals = vals_at  / dict_max_pot[row["ID_RMC"]] 
    new_row = row.values.copy()
    new_row[years_indices] = nvals
    new_rows.append(new_row)


dff = pd.DataFrame(new_rows, columns=cols)

dff["Outcome"] = "Performance"

dff.to_csv(home + "/data/clean/Outcomes/coneval_perf.csv", encoding='utf-8-sig', index=False)
    
