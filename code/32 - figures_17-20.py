import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import policy_priority_inference as ppi
from joblib import Parallel, delayed
import matplotlib.patches as mpatches

home =  os.getcwd()[:-4]

##################################################
##################################################
#
# Creates figures 17 to 20 (section 4.3)
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
sample_size = 10000
B = df_exp[colYears].values
B_dict = dict([(i,[i]) for i in range(N)])



B_wop = pd.read_csv(home+'/data/sims/optimal_B_wop.csv').values
B_flat = pd.read_csv(home+'/data/sims/optimal_B_flat.csv').values

B = df_exp[colYears].values

df_patches = df_id[["Derecho Social o Bienestar Económico (directo)", "Colour" ]].copy().drop_duplicates().reset_index(drop=True)
list_patches = [mpatches.Patch(color=df_patches.loc[i,"Colour"], label=df_patches.loc[i,"Derecho Social o Bienestar Económico (directo)"]) for i in range(len(df_patches))]





plt.figure(figsize=(8,4.5))
sorted_indices = df_id.sort_values('Derecho Social o Bienestar Económico (directo)').index.values
labels = []
all_diffs = (B_wop - B).mean(axis=1)
plt.plot(-100, -10000, '.', mec='w', mfc='k', markersize=20, label='year-specific')
plt.plot(-100, -10000, 's', mec='w', mfc='k', markersize=7, label='average')
for i, index in enumerate(np.argsort(all_diffs)):
    differences = B_wop[index,:] - B[index,:]
    for j, diff in enumerate(differences):
        plt.plot( i, diff, '.', mfc=df_id.loc[index,"Colour"], mec='w', markersize=1.5*(j+1)+5, alpha=.5)
    plt.plot( i, differences.mean(), 's', mfc=df_id.loc[index,"Colour"], mec='w', markersize=7)    
    labels.append(df_id.iloc[index].Label)
plt.xlim(-1, i+1)
plt.ylim(4, 22)
plt.ylabel('expenditure increase (pesos pc)', fontsize=14)
plt.gca().set_xticks(range(i+1))
plt.gca().set_xticklabels(labels, fontsize="x-small", rotation="vertical" )
plt.gca().spines[['right', 'top']].set_visible(False)
l1 = plt.legend(fontsize=10, loc=2)
plt.gca().add_artist(l1)
l2 = plt.legend(handles=list_patches, fontsize=10, loc=1, ncol=2)
plt.gca().add_artist(l2)
plt.tight_layout()
plt.savefig(home+'figures/figure_17a.pdf')
plt.show()





plt.figure(figsize=(8,4.5))
sorted_indices = df_id.sort_values('Derecho Social o Bienestar Económico (directo)').index.values
labels = []
all_diffs = (B_flat - B).mean(axis=1)
plt.plot(-100, -10000, '.', mec='w', mfc='k', markersize=20, label='year-specific')
plt.plot(-100, -10000, 's', mec='w', mfc='k', markersize=7, label='average')
for i, index in enumerate(np.argsort(all_diffs)):
    differences = B_flat[index,:] - B[index,:]
    for j, diff in enumerate(differences):
        plt.plot( i, diff, '.', mfc=df_id.loc[index,"Colour"], mec='w', markersize=1.5*(j+1)+5, alpha=.5)
    plt.plot( i, differences.mean(), 's', mfc=df_id.loc[index,"Colour"], mec='w', markersize=7)    
    labels.append(df_id.iloc[index].Label)
plt.xlim(-1, i+1)
plt.ylim(4, 22)
plt.ylabel('expenditure increase (pesos pc)', fontsize=14)
plt.gca().set_xticks(range(i+1))
plt.gca().set_xticklabels(labels, fontsize="x-small", rotation="vertical" )
plt.gca().spines[['right', 'top']].set_visible(False)
l1 = plt.legend(fontsize=10, loc=2)
plt.gca().add_artist(l1)
l2 = plt.legend(handles=list_patches, fontsize=10, loc=1, ncol=2)
plt.gca().add_artist(l2)
plt.tight_layout()
plt.savefig(home+'figures/figure_17b.pdf')
plt.show()








plt.figure(figsize=(8,4.5))
for right, group in df_id.groupby(by='Derecho Social o Bienestar Económico (directo)'):
    indices = group.index.values
    serie = np.sum(B_wop[indices,:] - B[indices,:], axis=0)
    plt.plot(colYearsInt, serie, linewidth=3, color=group.Colour.values[0])
    plt.plot(colYearsInt, serie, '.', markersize=20, color=group.Colour.values[0])
plt.gca().spines[['right', 'top']].set_visible(False)
plt.ylabel('additional funds (pesos pc)', fontsize=14)
plt.xlabel('year', fontsize=14)
plt.tight_layout()
plt.savefig(home+'figures/figure_18a.pdf')
plt.show()




plt.figure(figsize=(8,4.5))
for right, group in df_id.groupby(by='Derecho Social o Bienestar Económico (directo)'):
    indices = group.index.values
    serie = np.sum(B_flat[indices,:] - B[indices,:], axis=0)
    plt.plot(colYearsInt, serie, linewidth=3, color=group.Colour.values[0])
    plt.plot(colYearsInt, serie, '.', markersize=20, color=group.Colour.values[0])
plt.gca().spines[['right', 'top']].set_visible(False)
plt.ylabel('additional funds (pesos pc)', fontsize=14)
plt.xlabel('year', fontsize=14)
plt.tight_layout()
plt.savefig(home+'figures/figure_18b.pdf')
plt.show()




print('Baseline...')
Bs = get_dirsbursement_schedule(B, B_dict, T)
outputs = Parallel(n_jobs=5)(delayed(ppi.run_ppi)(I0, alpha, alpha_prime, beta, A=A, qm=qm, rl=rl, 
                    Imax=Imax, Imin=Imin, Bs=Bs, B_dict=B_dict) for sample in range(sample_size))
Is = np.array([output[0] for output in outputs])
IF0 = Is.mean(axis=0)[:,-1]

print('Flat...')
Bs = get_dirsbursement_schedule(B_flat, B_dict, T)
outputs = Parallel(n_jobs=5)(delayed(ppi.run_ppi)(I0, alpha, alpha_prime, beta, A=A, qm=qm, rl=rl, 
                    Imax=Imax, Imin=Imin, Bs=Bs, B_dict=B_dict) for sample in range(sample_size))
Is = np.array([output[0] for output in outputs])
IFF = Is.mean(axis=0)[:,-1]

print('Youth...')
Bs = get_dirsbursement_schedule(B_wop, B_dict, T)
outputs = Parallel(n_jobs=5)(delayed(ppi.run_ppi)(I0, alpha, alpha_prime, beta, A=A, qm=qm, rl=rl, 
                    Imax=Imax, Imin=Imin, Bs=Bs, B_dict=B_dict) for sample in range(sample_size))
Is = np.array([output[0] for output in outputs])
IFY = Is.mean(axis=0)[:,-1]

print('Random...')
Bs = get_dirsbursement_schedule(B*1.1, B_dict, T)
outputs = Parallel(n_jobs=5)(delayed(ppi.run_ppi)(I0, alpha, alpha_prime, beta, A=A, qm=qm, rl=rl, 
                    Imax=Imax, Imin=Imin, Bs=Bs, B_dict=B_dict) for sample in range(sample_size))
Is = np.array([output[0] for output in outputs])
IFR = Is.mean(axis=0)[:,-1]



plt.figure(figsize=(8,4.5))
sorted_indices = df_id.sort_values('Derecho Social o Bienestar Económico (directo)').index.values
labels = []
plt.plot(-100, -10000, '.', mec='w', mfc='k', markersize=20, label='no constraints')
plt.plot(-100, -10000, '*', mec='k', mfc='k', markersize=10, label='youth prioritisation')
for i, index in enumerate(sorted_indices):
    if IFY[index] > IFF[index]:
        plt.fill_between([i-.5, i+.5], [-10, -10], [100, 100], color='grey', alpha=50*(IFY[index]-IFF[index]))
    plt.plot(i, 100*(IFF[index]-IF0[index]), '.', mfc=df_id.loc[index,"Colour"], mec='w', markersize=20, alpha=1.0)
    plt.plot(i, 100*(IFY[index]-IF0[index]), '*', mfc=df_id.loc[index,"Colour"], mec='w', markersize=15, alpha=1.0)
    labels.append(df_id.iloc[index].Label)
plt.xlim(-1, i+1)
plt.ylim(-.5, 26)
plt.ylabel('additional percent coverage', fontsize=14)
plt.gca().set_xticks(range(i+1))
plt.gca().set_xticklabels(labels, fontsize="x-small", rotation="vertical" )
plt.gca().spines[['right', 'top']].set_visible(False)
l1 = plt.legend(fontsize=10, loc=1)
plt.gca().add_artist(l1)
l2 = plt.legend(handles=list_patches, fontsize=10, loc=2, ncol=2)
plt.gca().add_artist(l2)
plt.tight_layout()
plt.savefig(home+'figures/figure_19.pdf')
plt.show()





counter_Is = []
for programme in range(N):
    counter_Is.append(pd.read_csv(home+'/data/sims/sensitivity/'+str(programme)+'.csv').values[programme])
counter_Is = np.array(counter_Is)
IFI = counter_Is[:,-1]





plt.figure(figsize=(8,4.5))
sorted_indices = df_id.sort_values('Derecho Social o Bienestar Económico (directo)').index.values
labels = []
plt.plot(-100, -10000, '*', mec='k', mfc='k', markersize=10, label='coordinated')
plt.plot(-100, -10000, '^', mec='w', mfc='k', markersize=10, label='uncoordinated')
for i, index in enumerate(sorted_indices):
    plt.plot(i, 100*(IFR[index]-IF0[index])/B.sum(axis=1)[index]*.1, '^', mfc=df_id.loc[index,"Colour"], mec='w', markersize=10, alpha=1)
    plt.plot(i, 100*(IFF[index]-IF0[index])/(B_flat.sum(axis=1)[index] - B.sum(axis=1)[index]), '*', mfc=df_id.loc[index,"Colour"], mec='w', markersize=15, alpha=1)
    labels.append(df_id.iloc[index].Label)
plt.xlim(-1, i+1)
plt.ylim(-.01, .12)
plt.ylabel('additional percent coverage\n per additional pesos per capita', fontsize=14)
plt.gca().set_xticks(range(i+1))
plt.gca().set_xticklabels(labels, fontsize="x-small", rotation="vertical" )
plt.gca().spines[['right', 'top']].set_visible(False)
l1 = plt.legend(fontsize=10, loc=1)
plt.gca().add_artist(l1)
l2 = plt.legend(handles=list_patches, fontsize=10, loc=2, ncol=2)
plt.gca().add_artist(l2)
plt.tight_layout()
plt.savefig(home+'figures/figure_20.pdf')
plt.show()














