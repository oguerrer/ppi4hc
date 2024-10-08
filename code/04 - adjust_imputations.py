import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
warnings.simplefilter("ignore")

home =  os.getcwd()[0:-4]

###########################################################################
###########################################################################
#
# Corrects imputations in case they surpassed technical bounds
#
###########################################################################
###########################################################################


df = pd.read_csv(home+"/data/clean/Outcomes/coneval_perf_imputed.csv", encoding='utf-8-sig')

colYears = [col for col in df.columns if str(col).isnumeric()]
years = np.array([int(col) for col in df.columns if str(col).isnumeric()])
years_indices = df.columns.isin(colYears)

dfn = pd.read_csv(home+"/data/clean/Outcomes/coneval_perf.csv", encoding='utf-8-sig')


new_rows = []
for index in df.index.values:

    row_norm = dfn.loc[index]
    row_imputed = df.loc[index]
    observations = np.where(~row_norm[colYears].isnull())[0]
    missing_values = np.where(row_norm[colYears].isnull())[0]
    new_row = row_imputed.values.copy()
    vals = row_imputed[colYears].values.copy()
    
    if np.sum(row_imputed[colYears].isnull()) == 0 and len(missing_values) > 0:
        
        first_observation = observations[0]
        last_observation = observations[-1]
        
        vv = vals[first_observation:last_observation+1]
        vv = np.abs(vv[1::] - vv[0:-1])
        if np.sum(vv==0)==len(vv):
            vv = 10e-12
        
        vvf = vals[0:first_observation+1]
        vvf = np.abs(vvf[1::] - vvf[0:-1])
        
        vvl = vals[last_observation::]
        vvl = np.abs(vvl[1::] - vvl[0:-1])
        
        # check that there are missing values before the first observation
        if len(vvf) > 0:
            
            first_more_than_ub = np.sum(vals[0:first_observation]>1) > 0
            first_less_than_lb = np.sum(vals[0:first_observation]<0) > 0
            first_bigger_jump = np.max(vvf) > np.max(vv)
            
            if first_bigger_jump or first_more_than_ub or first_less_than_lb:

                if vals[first_observation] == 0:
                    vals[first_observation] = 10e-12
                if vals[first_observation] == 1:
                    vals[first_observation] = 1-10e-12
                ref_val = vals[first_observation]
                
                while first_bigger_jump or first_more_than_ub or first_less_than_lb:
                    
                    diff = 0.999*(vals - ref_val)
                    vals[0:first_observation] = ref_val + diff[0:first_observation]
                    vvf = vals[0:first_observation+1]
                    vvf = np.abs(vvf[1::] - vvf[0:-1])
                    
                    first_more_than_ub = np.sum(vals[0:first_observation]>1) > 0
                    first_less_than_lb = np.sum(vals[0:first_observation]<0) > 0
                    first_bigger_jump = np.max(vvf) > np.max(vv)
                

        # check that there are missing values after the last observation
        if len(vvl) > 0:
            
            last_more_than_ub = np.sum(vals[last_observation+1::]>1) > 0
            last_less_than_lb = np.sum(vals[last_observation+1::]<0) > 0
            last_bigger_jump = np.max(vvl) > np.max(vv)

            if last_bigger_jump or last_more_than_ub or last_less_than_lb:

                if vals[last_observation] == 0:
                    vals[last_observation] = 10e-12
                if vals[last_observation] == 1:
                    vals[last_observation] = 1-10e-12
                ref_val = vals[last_observation]

                while last_bigger_jump or last_more_than_ub or last_less_than_lb:
                    
                    diff = 0.999*(vals - ref_val)
                    vals[last_observation+1::] = ref_val + diff[last_observation+1::]
                    vvl = vals[last_observation::]
                    vvl = np.abs(vvl[1::] - vvl[0:-1])
                    
                    last_more_than_ub = np.sum(vals[last_observation+1::]>1) > 0
                    last_less_than_lb = np.sum(vals[last_observation+1::]<0) > 0
                    last_bigger_jump = np.max(vvl) > np.max(vv)



    vals[vals>1] = 1
    new_row[years_indices] = vals
    new_rows.append(new_row)


dff = pd.DataFrame(new_rows, columns=df.columns)

dff.to_csv(home + "/data/clean/Outcomes/coneval_perf_corrected.csv", encoding='utf-8-sig', index=False)




