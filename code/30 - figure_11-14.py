import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from joblib import Parallel, delayed
import policy_priority_inference as ppi



home =  os.getcwd()[:-4]

##################################################
##################################################
#
# Creates figures 11 to 14 (section 4.1)
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

# Budget
sample_size = 100
B = df_exp[colYears].values
B_dict = dict([(i,[i]) for i in range(N)])
Bs = get_dirsbursement_schedule(B, B_dict, T)

baseline = np.array(Parallel(n_jobs=20)(delayed(ppi.run_ppi)(I0, alpha, alpha_prime, beta, 
                                                             A=A, qm=qm, rl=rl, 
                    Imax=Imax, Imin=Imin, Bs=Bs, B_dict=B_dict) for sample in range(1000)))
Is_baseline = baseline.mean(axis=0)[0]
IF = Is_baseline[:,-1]









counter_Is = {}
for programme in range(N):
    counter_Is[programme] = pd.read_csv(home+'/data/sims/sensitivity/'+str(programme)+'.csv').values[programme]


elasticities = []
for programme in range(N):
    elasticity = counter_Is[programme] - IF[programme]
    elasticities.append(elasticity[-1])


df_patches = df_id[["Derecho Social o Bienestar Económico (directo)", "Colour" ]].copy().drop_duplicates().reset_index(drop=True)

list_patches = [mpatches.Patch(color=df_patches.loc[i,"Colour"], label=df_patches.loc[i,"Derecho Social o Bienestar Económico (directo)"]) for i in range(len(df_patches))]

colors = df_id.Colour.values








plt.figure(figsize=(8,4.5))
xdf = pd.DataFrame(zip(df_id['Derecho Social o Bienestar Económico (directo)'].values, np.array(elasticities)*100, IF*100), columns=['label', 'elasticity', 'potpop']).groupby('label').mean()
cd = dict(df_id[['Derecho Social o Bienestar Económico (directo)', 'Colour']].values)
for label, row in xdf.iterrows():
    if label == 'Social Protection':
        plt.plot(row.potpop, row.elasticity, '.', 
             markersize=10*np.sum(df_id.Colour==cd[label]), mfc=cd[label], mec='w', zorder=0)
    else:
        plt.plot(row.potpop, row.elasticity, '.', 
             markersize=10*np.sum(df_id.Colour==cd[label]), mfc=cd[label], mec='w', zorder=-1)
plt.ylabel('additional percent coverage', fontsize=14)
plt.xlabel('average percent coverage in 2022', fontsize=14)
plt.legend(handles=list_patches, fontsize=9, loc=2)
plt.xlim(1, 35)
plt.ylim(-.2, 2.02)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.tight_layout()
plt.savefig(home+'figures/figure_11.pdf')
plt.show()





plt.figure(figsize=(8,4.5))
coverages = 100*np.array(elasticities)
sorted_indices = np.argsort(coverages)
ticks_labels = df_id.Label.values[sorted_indices]
for i, index in enumerate(sorted_indices):
    coverage = coverages[index]
    plt.plot([i, i], [0, coverage], '--', linewidth=.5, mfc=colors[index])
    plt.plot([i],[coverage], '.', markersize=17, mfc=colors[index], mec='w')
plt.ylabel('additional percent coverage', fontsize=14)
plt.xticks(range(N), ticks_labels, fontsize="x-small", rotation="vertical" )
plt.legend(handles=list_patches, fontsize=9, loc=2)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.tight_layout()
plt.savefig(home+'figures/figure_12.pdf')
plt.show()












plt.figure(figsize=(8,4.5))
coverages = np.array(elasticities) * 100
covered_pop = IF * 100
sorted_indices = np.arange(N)
labels = df_id.Label.values
for i, index in enumerate(sorted_indices):
    coverage = coverages[index]
    plt.plot(covered_pop[index], coverage, '.', markersize=17, mfc=colors[index], mec='w')
    if coverages[index] > 1:
        plt.text(covered_pop[index], coverage, ' '+labels[index], rotation=45)
plt.ylim(-1, 10)
plt.ylabel('additional percent coverage', fontsize=14)
plt.xlabel('percent coverage in 2022', fontsize=14)
plt.legend(handles=list_patches, fontsize=9, loc=1, ncol=2)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.tight_layout()
plt.savefig(home+'figures/figure_13.pdf')
plt.show()

















plt.figure(figsize=(8,4.5))
sol_cols = ['Stage Of Life_'+str(i) for i in range(1,6)]
groupo_labels = ['Children', 'Youths', 'Adults18-30', 'Adults31+', 'Elderly'][::-1]
groupo_names = ['Children', 'Youths', 'Young Adults', 'Middle-age Adults', 'Elderly'][::-1]
markers = {'Adults18-30':'P', 'Adults31+':'d', 'Children':'^',
           'Elderly':'p', 'Youths':'*'}
for i, demogroup in enumerate(groupo_labels):
    marker = markers[demogroup]
    plt.scatter(-100, -100, marker=marker, color='k', label=demogroup)
for index, row in df_patches.iterrows():
    plt.scatter(-100, -100, marker='s', color=row.Colour, label=row['Derecho Social o Bienestar Económico (directo)'])
for i, demogroup in enumerate(groupo_names):
    qualified = np.sum(df_id[sol_cols].values == demogroup, axis=1) > 0
    xdf = pd.DataFrame(zip(df_id['Derecho Social o Bienestar Económico (directo)'].values, np.array(elasticities)*100, IF*100), columns=['label', 'elasticity', 'potpop'])[qualified].groupby('label').mean()
    cd = dict(df_id[['Derecho Social o Bienestar Económico (directo)', 'Colour']].values)
    for label, row in xdf.iterrows():
        plt.plot(row.potpop, row.elasticity, marker=markers[groupo_labels[i]], markersize=20, mfc=cd[label], mec='w')
plt.ylabel('additional percent coverage', fontsize=14)
plt.xlabel('average percent coverage in 2022', fontsize=14)
plt.legend( fontsize=9, ncol=2, loc=2)
plt.ylim(-.1, 2.5)
plt.xlim(-1, 40)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.tight_layout()
plt.savefig(home+'figures/figure_14.pdf')
plt.show()














