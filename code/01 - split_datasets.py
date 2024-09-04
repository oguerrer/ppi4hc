import os, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

home =  os.getcwd()[0:-4]

#########################
#########################
#
# Combines raw data
#
#########################
#########################

# Pivot dataset, 2 categories: 1) Exp, 2) Outcomes
# 3 Exp datasets: 1) Original, 2) Modificado , 3) Ejercido
# 5 Outcomes datasets: 1) Pob Atendida, 2) Pob Objetivo, 3) Pob Potencial

df_full = pd.read_csv(home + '/data/raw/coneval_to_impute.csv', encoding='utf-8-sig')

to_keep = ["ID_RMC", "Ciclo"]
to_keep_small = ["ID_RMC",]

list_exp = ["Original", "Modificado", "Ejercido"]

df_exp_melt = df_full.melt(id_vars=to_keep, value_vars=list_exp, var_name="Presupuesto") 
df_exp_pivot = df_exp_melt.pivot(index= to_keep_small + ["Presupuesto"], columns="Ciclo", values="value").reset_index()

df_exp_pivot.to_csv(home + "/data/clean/Expenditure/coneval_exp_pivot.csv", encoding='utf-8-sig', index=False)

# Exclude coverage and effectiveness

list_oc = ["Población Potencial", "Población Objetivo", "Población Atendida"]
df_oc_melt = df_full.melt(id_vars=to_keep, value_vars=list_oc, var_name="Outcome") 
df_oc_pivot = df_oc_melt.pivot(index= to_keep_small + ["Outcome"], columns="Ciclo", values="value").reset_index()
df_oc_pivot.to_csv(home + "/data/clean/Outcomes/coneval_oc_pivot.csv", encoding='utf-8-sig', index=False)

