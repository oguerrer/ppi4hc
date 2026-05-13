import os
import pandas as pd
import numpy as np
import policy_priority_inference as ppi
from joblib import Parallel, delayed
from tqdm import tqdm

home =  os.getcwd()[:-4]

np.random.seed(0) 

##################################################
##################################################
#
# Runs baseline simulations
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

colYears = [col for col in df_var.columns if col.isnumeric()]
colYearsInt = [int(col) for col in df_var.columns if col.isnumeric()]


# Indicators
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


sample_size = 100000


# Budget
B = df_exp[colYears].values
B_dict = dict([(i,[i]) for i in range(N)])
Bs = get_dirsbursement_schedule(B, B_dict, T)

baseline = np.array(
    Parallel(n_jobs=-1)(
        delayed(ppi.run_ppi)(
            I0, alpha, alpha_prime, beta,
            A=A, qm=qm, rl=rl,
            Imax=Imax, Imin=Imin, Bs=Bs, B_dict=B_dict
        )
        for sample in tqdm(range(sample_size), desc="Baseline", total=sample_size)
    )
)

Is_baseline = baseline.mean(axis=0)[0]
IF = Is_baseline[:,-1]


dff = pd.DataFrame(Is_baseline, columns=[str(i) for i in range(Bs.shape[-1])])
dff.to_csv(home+'/data/sims/baseline.csv', index=False)






















