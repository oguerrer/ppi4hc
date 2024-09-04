import os
import pandas as pd
import numpy as np
import policy_priority_inference as ppi
from joblib import Parallel, delayed

home =  os.getcwd()[:-4]


##################################################
##################################################
#
# Runs optimization required for section 4.3
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


# Choose between 'flat' or 'weighted'. Flat is used to optimize without constraints
# other than maintaining the budget balance. With weights, the algorithm gives priority
# to programmes that cover the youth.
opt_type = 'flat'

df_var = pd.read_csv(home + '/data/clean/Variables/add_variables.csv', encoding='utf-8-sig')
df_oc = pd.read_csv(home + '/data/clean/Outcomes/coneval_perf_corrected.csv', encoding='utf-8-sig')
df_exp = pd.read_csv(home + '/data/clean/Expenditure/coneval_exp_detrend.csv', encoding='utf-8-sig')
df_par = pd.read_csv(home + '/data/clean/parameters.csv', encoding='utf-8-sig')
df_id = pd.read_excel(home + '/data/clean/Outcomes/coneval_labels.xlsx', sheet_name="Sheet1")

colYears = [col for col in df_var.columns if col.isnumeric()]
colYearsInt = [int(col) for col in df_var.columns if col.isnumeric()]


rights_dict = dict([(right, group.index.values) for right, group in df_id.groupby('Derecho Social o Bienestar Económico (directo)')])


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


shares = B.sum(axis=1)/B.sum()
to_increase = np.where(shares < .05)[0]
n = len(to_increase)
extra_budget = B.sum() * .1



# Optional weights according to age group
group_names = ['Children', 'Youths', 'Young Adults', 'Middle-age Adults', 'Elderly']
weights_groups = dict(zip(group_names, [16/31, 8/31, 4/31, 2/31, 1/31]))
age_weights = np.zeros(N)
for index, row in df_id.iterrows():
    groups = row[['Stage Of Life_1','Stage Of Life_2', 'Stage Of Life_3', 'Stage Of Life_4', 'Stage Of Life_5']].dropna()
    meanweight = np.mean([weights_groups[group] for group in groups])
    age_weights[index] = meanweight



# Objective function to be optimized
def run_par(I0, alpha, alpha_prime, beta, A, qm, rl, Imax, Imin, B, B_dict, solution):
    sol = B.copy()
    sol += solution.reshape(N, B.shape[1]) * extra_budget
    
    Bs = get_dirsbursement_schedule(sol, B_dict, T)
    outputs = np.array(Parallel(n_jobs=15)(delayed(ppi.run_ppi)(I0, alpha, alpha_prime, beta, A=A, qm=qm, rl=rl, 
                          Imax=Imax, Imin=Imin, Bs=Bs, B_dict=B_dict) for i in range(sample_size)))
    Is = np.array([output[0] for output in outputs]).mean(axis=0)
    if opt_type == 'flat':
        return np.average( Is[:,-1], weights=None ), sol
    else:
        return np.average( Is[:,-1], weights=age_weights ), sol
    





Bs = get_dirsbursement_schedule(B, B_dict, T)
outputs = Parallel(n_jobs=60)(delayed(ppi.run_ppi)(I0, alpha, alpha_prime, beta, A=A, qm=qm, rl=rl, 
                    Imax=Imax, Imin=Imin, Bs=Bs, B_dict=B_dict) for sample in range(sample_size))
Is = np.array([output[0] for output in outputs]).mean(axis=0)
if opt_type == 'flat':
    best_fitness = np.average(Is[:,-1], weights=None)
else:
    best_fitness = np.average(Is[:,-1], weights=age_weights)



# Differential evolution algorithm
popsize = 24
njobs = 2
mut=0.08
crossp=0.7

TB = len(colYears)*N
bounds = np.array(list(zip(.0001*np.ones(TB), .99*np.ones(TB))))
min_b, max_b = np.asarray(bounds).T
diff = np.fabs(min_b - max_b)
dimensions = len(bounds)
pop =  np.random.rand(popsize, dimensions)*.8 + .2
for i in range(popsize):
    pop[i] /= pop[i].sum()
pop[0] = B.flatten()/B.sum()
outputs = []

for step in range(100000):
    
    print(step)
    
    results = [run_par(I0, alpha, alpha_prime, beta, A, qm, rl, Imax, Imin, B, B_dict, solution) for solution in pop]
    fitness, BB = zip(*results)
    fitness = np.array(fitness)
    best_idx = np.argmax(fitness)
    best_Bsol = BB[best_idx]
    
    if fitness[best_idx] > best_fitness:
        best_sol = pop[best_idx]
        best_fitness = fitness[best_idx]
        print(best_fitness, best_idx)
        
        df_sol = pd.DataFrame(best_Bsol, columns=colYears)
        if opt_type == 'flat':
            df_sol.to_csv(home+'/data/sims/optimal_B_flat.csv', index=False)
        elif opt_type == 'weighted':
            df_sol.to_csv(home+'/data/sims/optimal_B_wop.csv', index=False)

    sorter = np.argsort(fitness)[::-1]
    survivors = pop[sorter][0:int(len(fitness)/2)].tolist()
    new_pop = survivors.copy()
    
    newPop = []
    for j in range(len(survivors)):
        idxs = [idx for idx in range(len(survivors)) if idx != j]
        a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
        mutant = np.clip(a + mut * (b - c), 0, 1)
        cross_points = np.random.rand(dimensions) < crossp
        if not np.any(cross_points):
            cross_points[np.random.randint(0, dimensions)] = True
        trial = np.where(cross_points, mutant, pop[j])
        trial_denorm = min_b + trial * diff
        new_pop.append(trial_denorm)
        
    pop = np.array(new_pop)
    for i in range(popsize):
        pop[i] /= pop[i].sum()






















