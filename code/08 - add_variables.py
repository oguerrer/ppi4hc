import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

home =  os.getcwd()[0:-4]


##################################################
##################################################
#
# Integrate file with supporting data
#
##################################################
##################################################


df = pd.read_csv(home+"/data/clean/Outcomes/coneval_perf_corrected.csv", encoding='utf-8-sig')
colYears = [col for col in df.columns if str(col).isnumeric()]

rl = pd.read_csv(home+"/data/clean/Variables/governance_rl_normalized.csv")
cc = pd.read_csv(home+"/data/clean/Variables/governance_cc_normalized.csv")

mex_rl = rl.loc[rl.countryCode=="MEX", colYears].copy()
mex_cc = cc.loc[cc.countryCode=="MEX", colYears].copy()

cpi = pd.read_csv(home+"/data/raw/cpi.csv")
pop = pd.read_csv(home+"/data/raw/population.csv")

mex_cpi = cpi.loc[cpi.CountryCode=="MEX", colYears].copy()
mex_pop = pop.loc[pop.CountryCode=="MEX", colYears].copy()

rows = [["CC"] + list(mex_cc.values[0]), ["RL"] +list(mex_rl.values[0]), ["CPI"] + list(mex_cpi.values[0]), ["POP"] +list(mex_pop.values[0])]

variables = ["Variable"] + colYears
dff = pd.DataFrame(rows, columns = variables)

dff.to_csv(home+"/data/clean/Variables/add_variables.csv", index=False)
