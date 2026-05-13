import os
import pandas as pd
import numpy as np
import policy_priority_inference as ppi
from joblib import Parallel, delayed
import matplotlib.patches as mpatches
from tqdm import tqdm

home =  os.getcwd()[:-4]

np.random.seed(0) 

##################################################
##################################################
#
# Runs scenarios with optimised budgets
#
##################################################
##################################################





def get_dirsbursement_schedule(Bs, B_dict, T):
    programs = sorted(list(set([item for subl in list(B_dict.values()) for item in subl])))
    B_sequence = [[] for program in programs]
    subperiods = int(T/Bs.shape[1])
    for i, program in enumerate(programs):
        for period in range(Bs.shape[1]):
            for subperiod in range(subperiods):
                B_sequence[i].append( Bs[i,period]/subperiods )
    B_sequence = np.array(B_sequence)
    return B_sequence





df_var = pd.read_csv(home + '/data/clean/Variables/add_variables.csv', encoding='utf-8-sig')
df_oc = pd.read_csv(home + '/data/clean/Outcomes/coneval_perf_corrected.csv', encoding='utf-8-sig')
df_exp = pd.read_csv(home + '/data/clean/Expenditure/coneval_exp_detrend.csv', encoding='utf-8-sig')
df_par = pd.read_csv(home + '/data/clean/parameters.csv', encoding='utf-8-sig')
df_id = pd.read_excel(home + '/data/clean/Outcomes/coneval_labels.xlsx', sheet_name="Sheet1")

colYears = [col for col in df_var.columns if col.isnumeric()]
colYearsInt = [int(col) for col in df_var.columns if col.isnumeric()]

series = df_oc[colYears].values
N = len(df_oc)
I0 = series[:,0]
sub_periods = 6
T = len(colYears)*sub_periods
Imax = np.ones(N)
Imin = np.zeros(N)


qm = np.ones(N)*np.mean(df_var.loc[df_var.Variable=="CC",:][colYears].values)
rl = np.ones(N)*np.mean(df_var.loc[df_var.Variable=="RL",:][colYears].values)
alpha = df_par.alpha
beta = df_par.beta
alpha_prime = df_par.alpha_prime


df_net = pd.read_csv(home+'/data/clean/network.csv')
A = np.zeros((N, N)) # adjacency matrix
for index, row in df_net.iterrows():
    i = int(row.From)
    j = int(row.To)
    w = row.Weight
    A[i,j] = w



# Budget
B = df_exp[colYears].values
B_dict = dict([(i,[i]) for i in range(N)])



B_wop = pd.read_csv(home+'/data/sims/optimal_B_weighted.csv').values
B_flat = pd.read_csv(home+'/data/sims/optimal_B_flat.csv').values

B = df_exp[colYears].values

df_patches = df_id[["Derecho Social o Bienestar Económico (directo)", "Colour" ]].copy().drop_duplicates().reset_index(drop=True)
list_patches = [mpatches.Patch(color=df_patches.loc[i,"Colour"], label=df_patches.loc[i,"Derecho Social o Bienestar Económico (directo)"]) for i in range(len(df_patches))]



B_dict = dict([(i,[i]) for i in range(N)])
Bs = get_dirsbursement_schedule(B, B_dict, T)

sample_size = 100000



print('Flat...')
Bs = get_dirsbursement_schedule(B_flat, B_dict, T)
outputs = Parallel(n_jobs=-1)(
    delayed(ppi.run_ppi)(
        I0, alpha, alpha_prime, beta,
        A=A, qm=qm, rl=rl,
        Imax=Imax, Imin=Imin, Bs=Bs, B_dict=B_dict
    )
    for sample in tqdm(range(sample_size), desc="Outputs", total=sample_size)
)
Is = np.array([output[0] for output in outputs])
IFF = Is.mean(axis=0)[:,-1]
pd.DataFrame([IFF]).to_csv(home+'/data/sims/IFF.csv', index=False)

print('Youth...')
Bs = get_dirsbursement_schedule(B_wop, B_dict, T)
outputs = Parallel(n_jobs=-1)(
    delayed(ppi.run_ppi)(
        I0, alpha, alpha_prime, beta,
        A=A, qm=qm, rl=rl,
        Imax=Imax, Imin=Imin, Bs=Bs, B_dict=B_dict
    )
    for sample in tqdm(range(sample_size), desc="Outputs", total=sample_size)
)
Is = np.array([output[0] for output in outputs])
IFY = Is.mean(axis=0)[:,-1]
pd.DataFrame([IFY]).to_csv(home+'/data/sims/IFY.csv', index=False)

print('Random...')
Bs = get_dirsbursement_schedule(B*1.1, B_dict, T)
outputs = Parallel(n_jobs=-1)(
    delayed(ppi.run_ppi)(
        I0, alpha, alpha_prime, beta,
        A=A, qm=qm, rl=rl,
        Imax=Imax, Imin=Imin, Bs=Bs, B_dict=B_dict
    )
    for sample in tqdm(range(sample_size), desc="Outputs", total=sample_size)
)
Is = np.array([output[0] for output in outputs])
IFR = Is.mean(axis=0)[:,-1]
pd.DataFrame([IFR]).to_csv(home+'/data/sims/IFR.csv', index=False)























