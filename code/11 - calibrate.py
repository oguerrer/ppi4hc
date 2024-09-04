import os
import pandas as pd
import numpy as np
import policy_priority_inference as ppi

home =  os.getcwd()[:-4]


#########################
#########################
#
# Calibrates PPI
#
#########################
#########################


# Helper functions
def get_success_rates(series):
    sc = series[:, 1::]-series[:, 0:-1] 
    success_rates = np.sum(sc>0, axis=1)/sc.shape[1] 
    success_rates = .9*(success_rates-success_rates.min())/(success_rates.max()-success_rates.min()) + .05
    return success_rates


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
df_id = pd.read_excel(home + '/data/clean/Outcomes/coneval_labels.xlsx', sheet_name="Sheet1")

df_oc['right'] = df_id['Derecho Social o Bienestar Económico (directo)']

colYears = [col for col in df_var.columns if col.isnumeric()]

df_oc["Instrumental"] = 1


# Indicators
series = df_oc[colYears].values
N = len(df_oc)
I0 = series[:,0]
IF = series[:,-1]

sub_periods = 6
T = len(colYears)*sub_periods

changes = series[:,1::] - series[:,0:-1]
rights_rates = dict([ (right, np.sum(changes[d.index]>0)/(changes[d.index].shape[0]*changes[d.index].shape[1])) for right, d in df_oc.groupby('right')])
success_rates = np.array([rights_rates[right] for right in df_oc.right.values])


# Governance
qm = np.ones(N)*np.mean(df_var.loc[df_var.Variable=="CC",:][colYears].values)
rl = np.ones(N)*np.mean(df_var.loc[df_var.Variable=="RL",:][colYears].values)


# Budget
B = df_exp[colYears].values
B_dict = dict([(i,[i]) for i in range(N)])
Bs = get_dirsbursement_schedule(B, B_dict, T)


# Network
df_net = pd.read_csv(home+'/data/clean/network.csv')
A = np.zeros((N, N)) # adjacency matrix
for index, row in df_net.iterrows():
    i = int(row.From)
    j = int(row.To)
    w = row.Weight
    A[i,j] = w

parameters = ppi.calibrate(I0, IF, success_rates, A=A, R=None, bs=None, qm=qm, rl=rl, 
          Bs=Bs, B_dict=B_dict, T=T, threshold=0.97, parallel_processes=20,
          verbose=True, low_precision_counts=50, increment=1000)

dff = pd.DataFrame(parameters[1::,:], columns=parameters[0])
dff.to_csv(home+'/data/clean/parameters.csv', index=False)








