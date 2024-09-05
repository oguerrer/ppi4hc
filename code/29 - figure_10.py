import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")


home =  os.getcwd()[:-4]

#########################
#########################
#
# Creates figure 10
#
#########################
#########################


df_var = pd.read_csv(home + '/data/clean/Variables/add_variables.csv', encoding='utf-8-sig')
df_oc = pd.read_csv(home + '/data/clean/Outcomes/coneval_perf_corrected.csv', encoding='utf-8-sig')
df_exp = pd.read_csv(home + '/data/clean/Expenditure/coneval_exp_detrend.csv', encoding='utf-8-sig')
df_par = pd.read_csv(home + '/data/clean/parameters.csv', encoding='utf-8-sig')
df_id = pd.read_excel(home + '/data/clean/Outcomes/coneval_labels.xlsx', sheet_name="Sheet1")

colYears = [col for col in df_var.columns if col.isnumeric()]
colYearsInt = [int(col) for col in df_var.columns if col.isnumeric()]

series = df_oc[colYears].values
N = len(df_oc)
Imax = np.ones(N)
IF = series[:,-1]

df_id["FinalCov"] = IF

df_trim = df_id[['Label', 'Derecho Social o Bienestar Económico (directo)', 'Colour',
                'Stage Of Life_1', 'Stage Of Life_2', 'Stage Of Life_3', 
                'Stage Of Life_4', 'Stage Of Life_5']].copy()

df_trim.set_index("Label", drop=True, inplace=True)

dict_id = df_trim.to_dict(orient="index")

for i in dict_id:
    dict_id[i]["Stage of Life"] = [dict_id[i]["Stage Of Life_1"], dict_id[i]["Stage Of Life_2"],
                                       dict_id[i]["Stage Of Life_3"], dict_id[i]["Stage Of Life_4"],
                                       dict_id[i]["Stage Of Life_5"]]
    
SoL = ['Children', 'Youths', 'Young Adults', 'Middle-age Adults', 'Elderly']
dict_SoL = { i : {"Programs" : [], "AvgFinalCov" : np.nan} for i in SoL }

for i in dict_SoL:
    for prog in dict_id:
        if i in dict_id[prog]["Stage of Life"]:
            dict_SoL[i]["Programs"].append(prog)

for i in dict_SoL:
    avg = []
    for prog in dict_SoL[i]["Programs"]:
        avg.append(df_id.loc[df_id.Label==prog,"FinalCov"].values[0])
    dict_SoL[i]["AvgFinalCov"] = np.mean(avg)
        

dict_SoL["Children"]["Colour"] = "#FEC896"
dict_SoL["Youths"]["Colour"] = "#FDA453"
dict_SoL["Young Adults"]["Colour"] = "#FD8619"
dict_SoL["Middle-age Adults"]["Colour"] = "#D86802"
dict_SoL["Elderly"]["Colour"] = "#AC5302"

# matplotlib.rcParams['font.sans-serif'] = "Garamond"
# matplotlib.rcParams['font.family'] = "sans-serif"



plt.figure(figsize=(8,4.5))
SoL_g = ['Children', 'Youths', 'Young Adults', 'Middle-age\n Adults', 'Elderly']
i=0
for age in SoL:
    plt.bar(i+1, round((dict_SoL[age]["AvgFinalCov"])*100,1), color=dict_SoL[age]["Colour"])
    plt.text(i+1, (dict_SoL[age]["AvgFinalCov"])*100 + .5, str(round(dict_SoL[age]["AvgFinalCov"]*100,1))+'%', 
             ha="center", va="center", fontsize=12,
             bbox=dict(facecolor='none', edgecolor='none', boxstyle='round', pad=.1, linewidth=.1))
    i+=1
plt.ylim(14, 22)
plt.gca().set_xticks(range(len(SoL)+1))
plt.ylabel('average coverage in 2022', fontsize=14)
plt.xlabel('age groups', fontsize=14)
plt.gca().set_xticklabels([""] +SoL_g, fontsize=12)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlim(.5, 5.5)
plt.tight_layout()
plt.savefig(home+'figures/figure_10.pdf')
plt.show()









