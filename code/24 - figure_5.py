import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]

#########################
#########################
#
# Creates figure 5
#
#########################
#########################

df22 = pd.read_excel(home+"/data/raw/IF2022_newlabels.xlsx", sheet_name="Inventario_2022")
newlab = pd.read_excel(home+"/data/clean/Outcomes/labels_merging.xlsx", sheet_name="Sheet1")


df22.rename({ 'ID_RMC1' : "ID_RMC"}, inplace=True, axis=1)
df22_small = df22[["ID_RMC", "SR", "SR_NEW", "Ejercido"]].copy()

for i, row in df22_small.iterrows():
    if row["ID_RMC"] in newlab.ID_RMC.values :
        df22_small.loc[i,"SR_NEW"] = newlab.loc[newlab.ID_RMC==row["ID_RMC"],:]["Social Right"].values[0]        
        
for i,row in df22_small.iterrows():
    df22_small.loc[i,"Ejercido"] = float(str(row["Ejercido"]).replace("$","").replace(",","").strip())

df22_small["Ejercido_Perc"] = 0
for i,row in df22_small.iterrows():
    df22_small.loc[i,"Ejercido_Perc"] = df22_small.loc[i,"Ejercido"] / np.sum(df22_small.Ejercido)

df22_small["SR_Perc"] = df22_small.groupby("SR_NEW")["Ejercido_Perc"].transform(np.sum)

df22_sr = df22_small[["SR_NEW", "SR_Perc"]].copy().drop_duplicates().reset_index(drop=True)


dict_col = {'Minorities & EDI':'#FFCCCC',
                'Social Protection':'#8EA9DB',
                'Education':'#C65911',
                'Working Conditions':'#FFC000',
                'Health':'#A9D08E',
    }

SR = [i for i in dict_col]

dict_label = {'Minorities & EDI':'Minorities\n& EDI',
                'Education':'Education',
                'Working Conditions':'Working\nConditions',
                'Health':'Health',
                'Social Protection':'Social\nProtection'
    }





df22_small["SR_Prop"] = df22_small.groupby("SR_NEW")["ID_RMC"].transform("count")
df22_small["SR_Prop"] = df22_small["SR_Prop"] / len(df22_small)
df22_sr_prop = df22_small[["SR_NEW", "SR_Prop"]].copy().drop_duplicates().reset_index(drop=True)





i=0
plt.figure(figsize=(8,4.5))
for sr in SR:
    plt.bar(i+1, round((df22_sr_prop.loc[df22_sr_prop.SR_NEW==sr,"SR_Prop"])*100,2), color=dict_col[sr])
    plt.text(i+1, (df22_sr_prop.loc[df22_sr_prop.SR_NEW==sr,"SR_Prop"])*100 + 2, str(round(df22_sr_prop.loc[df22_sr_prop.SR_NEW==sr,"SR_Prop"].values[0]*100,2))+'%', 
             ha="center", va="center", fontsize=12,
             bbox=dict(facecolor='none', edgecolor='none', boxstyle='round', pad=.1, linewidth=.1))
    i+=1
plt.ylim(0, 45)
plt.xlim(.5, 5.5)
plt.gca().set_xticks(range(1, len(SR)+1))
plt.ylabel('programs in 2022', fontsize=14)
plt.xlabel('policy area', fontsize=14)
plt.gca().set_xticklabels([dict_label[i] for i in SR], fontsize=10, rotation=45)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(home+'figures/figure_5.pdf')
plt.show()








