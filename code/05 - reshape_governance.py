import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

home =  os.getcwd()[0:-4]

##################################################
##################################################
#
# Reshape raw data on public governance indicators
#
##################################################
##################################################

dfcc = pd.read_excel(home+"/data/raw/wgidataset.xlsx", sheet_name='ControlofCorruption', skiprows=14)
dfrl = pd.read_excel(home+"/data/raw/wgidataset.xlsx", sheet_name='RuleofLaw', skiprows=14)


colYears = [str(c) for c in range(1996, 2001, 2)]+[str(c) for c in range(2002, 2023)]

relevant_columns = [c for c in dfcc.columns if 'Estimate' in c or 'Code' in c]

dfcc = pd.DataFrame(dfcc[relevant_columns].values, columns=['countryCode']+colYears)
dfrl = pd.DataFrame(dfrl[relevant_columns].values, columns=['countryCode']+colYears)


dfcc.to_csv(home+"/data/clean/Variables/governance_cc_reshaped.csv", index=False)
dfrl.to_csv(home+"/data/clean/Variables/governance_rl_reshaped.csv", index=False)


