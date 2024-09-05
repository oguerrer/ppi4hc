import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")
import matplotlib.patches as mpatches
import matplotlib


home =  os.getcwd()[:-4]

#########################
#########################
#
# Creates figure 6
#
#########################
#########################


df_var = pd.read_csv(home + '/data/clean/Variables/add_variables.csv', encoding='utf-8-sig')
df_oc = pd.read_csv(home + '/data/clean/Outcomes/coneval_perf_corrected.csv', encoding='utf-8-sig')
df_exp = pd.read_csv(home + '/data/clean/Expenditure/coneval_exp_detrend.csv', encoding='utf-8-sig')
df_par = pd.read_csv(home + '/data/clean/parameters.csv', encoding='utf-8-sig')
df_id = pd.read_excel(home + '/data/clean/Outcomes/coneval_labels.xlsx', sheet_name="Sheet1")

dft = df_id[["ID_RMC"]].copy()
dft["Data"] = 100
df_id.sort_values(by=['Derecho Social o Bienestar Económico (directo)'], inplace=True, ignore_index=True)

df_sect = df_id[['Derecho Social o Bienestar Económico (directo)']].copy()
df_sect["Count"] = 1
df_sect["N"] = df_sect.groupby('Derecho Social o Bienestar Económico (directo)')["Count"].transform(sum)
df_sect['Percent'] = df_sect["N"].values / 49
colYears = [col for col in df_var.columns if col.isnumeric()]
colYearsInt = [int(col) for col in df_var.columns if col.isnumeric()]

series = df_oc[colYears].values
N = len(df_oc)
I0 = series[:,0]
IF = series[:,-1]


df_patches = df_id[["Derecho Social o Bienestar Económico (directo)", "Colour" ]].copy().drop_duplicates().reset_index(drop=True)
list_patches = [mpatches.Patch(color=df_patches.loc[i,"Colour"], label=df_patches.loc[i,"Derecho Social o Bienestar Económico (directo)"]) for i in range(len(df_patches))]

matplotlib.rcParams['font.sans-serif'] = "Garamond"
matplotlib.rcParams['font.family'] = "sans-serif"
 
        
fig=plt.figure(figsize=(5.5,5.5))
ax = plt.subplot(111, polar=True)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

data = dft['Data'].values.copy()

    
N = len(data)
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
width = 2 * np.pi / N
bars = ax.bar(theta, data, width=width, bottom=0.0)

ax.xaxis.set_ticklabels([])
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.yaxis.set_ticklabels([])
ax.spines["polar"].set_visible(True)
ax.grid(False)
for i, bar in enumerate(bars):

    rotation = 90-theta[i]*180/np.pi
    ha, va = 'center', 'center'
    bar.set_facecolor(df_id.loc[i,"Colour"])
    plt.text((theta[i]), data[i]-99.1, df_id.loc[i,"Label"], 
             rotation=rotation, ha=ha, va=va, fontsize=7,
             bbox=dict(facecolor='none', edgecolor='none', boxstyle='round', pad=.1, linewidth=.2))

plt.text(theta[7], 0.4, "33%", 
         ha=ha, va=va, fontsize=15,
         bbox=dict(facecolor='none', edgecolor='none', boxstyle='round', pad=.3, linewidth=.5))
plt.text(theta[23], 0.4, "31%", 
         ha=ha, va=va, fontsize=15,
         bbox=dict(facecolor='none', edgecolor='none', boxstyle='round', pad=.3, linewidth=.5))
plt.text(theta[35], 0.4, "18%", 
         ha=ha, va=va, fontsize=15,
         bbox=dict(facecolor='none', edgecolor='none', boxstyle='round', pad=.3, linewidth=.5))
plt.text(theta[41], 0.4, "6%", 
         ha=ha, va=va, fontsize=15,
         bbox=dict(facecolor='none', edgecolor='none', boxstyle='round', pad=.3, linewidth=.5))
plt.text(theta[46], 0.4, "12%", 
         ha=ha, va=va, fontsize=15,
         bbox=dict(facecolor='none', edgecolor='none', boxstyle='round', pad=.3, linewidth=.5))

plt.legend(handles=list_patches, fontsize=9, bbox_to_anchor=(1.2, 1.2))
plt.ylim(0, 1.1)
ax.axis("off")
plt.tight_layout()
plt.savefig(home+'figures/figure_6.pdf')
plt.show()

































































































































