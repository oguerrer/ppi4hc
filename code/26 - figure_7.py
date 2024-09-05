import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches


home =  os.getcwd()[:-4]

#########################
#########################
#
# Creates figure 7
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


df_exp_per = df_exp.copy()

for year in colYears:
    df_exp_per.loc[:, year] = (df_exp_per[year].values / df_exp_per[year].sum())*100

df_id_trim = df_id[['ID_RMC', 'Label', 'Derecho Social o Bienestar Económico (directo)', 'Colour']].copy()

df_final = df_exp_per.merge(df_id_trim, on = "ID_RMC")

for sec in np.unique(df_final[["Derecho Social o Bienestar Económico (directo)"]]):
    for year in colYears :
        df_final[str(year)+"_Sec"] = df_final.groupby("Derecho Social o Bienestar Económico (directo)")[year].transform(sum)
             

sectors = [i for i in np.unique(df_final[["Derecho Social o Bienestar Económico (directo)"]])]

df_final.rename({"Derecho Social o Bienestar Económico (directo)":"Sectors"}, axis=1, inplace=True)



df_patches = df_id[["Derecho Social o Bienestar Económico (directo)", "Colour" ]].copy().drop_duplicates().reset_index(drop=True)
list_patches = [mpatches.Patch(color=df_patches.loc[i,"Colour"], label=df_patches.loc[i,"Derecho Social o Bienestar Económico (directo)"]) for i in range(len(df_patches))]



for sec in sectors:
    for year in colYears :
        df_final[str(year)+"_Avg"] = df_final.groupby("Sectors")[year].transform(np.mean)

sectors = [i for i in np.unique(df_final[["Sectors"]])]

df_final_trim = df_final[['Sectors','2016_Avg', '2017_Avg', '2018_Avg', '2019_Avg',
                  '2020_Avg', '2021_Avg', '2022_Avg']].copy().drop_duplicates().set_index("Sectors",drop=True)

dict_exp = {sec : df_final_trim.loc[sec,:].values.tolist()for sec in sectors}

width = 0.15
x = np.arange(len(colYears))
multiplier = 0




plt.figure(figsize=(8,4.5))
for sec, perf in dict_exp.items():
    offset = width * multiplier
    rects = plt.gca().bar(x + offset, np.asarray(perf), 
            width, color=df_final.loc[df_final.Sectors==sec, "Colour"].values[0],
            )
    multiplier += 1
plt.gca().set_xticks(x + width*2, colYears)
plt.xlabel("year", fontsize=14)
plt.ylabel("average percent share of budget", fontsize=14)
legend_2 = plt.gca().legend(handles=list_patches, fontsize=10, loc='upper center', ncol=3)
plt.gca().add_artist(legend_2)
plt.ylim(0, 5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(home+"figures/figure_7.pdf")
plt.show()









