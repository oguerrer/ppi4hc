import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

# Set home to OneDrive ppi4hc
home =  os.getcwd()[:-4]

##################################################
##################################################
#
# Creates figures A1 and A2 from the appendix
#
##################################################
##################################################


df22 = pd.read_excel(home+"/data/raw/IF2022_adj.xlsx", sheet_name="Inventario_2022")

df22["Ramo ID"] = ""

for i,row in df22.iterrows():
    df22.loc[i,"Ramo ID"] = str(row["ID_RMC1"].split("-")[0])
    
ramos = df22[["Ramo ID", "Ramo"]].drop_duplicates()
df22.rename({ 'ID_RMC1' : "ID_RMC", "Derecho Social o Bienestar Económico (directo)" : "Social_Right"}, inplace=True, axis=1)
df22_small = df22[["ID_RMC", "Social_Right", "Ejercido"]].copy()
    
for i,row in df22_small.iterrows():
    df22_small.loc[i,"Ejercido"] = float(str(row["Ejercido"]).replace("$","").replace(",","").strip())

df22_small["Ejercido_Perc"] = 0
for i,row in df22_small.iterrows():
    df22_small.loc[i,"Ejercido_Perc"] = df22_small.loc[i,"Ejercido"] / np.sum(df22_small.Ejercido)

df22_small["SR_Perc"] = df22_small.groupby("Social_Right")["Ejercido_Perc"].transform(np.mean)

df22_sr = df22_small[["Social_Right", "SR_Perc"]].copy().drop_duplicates().reset_index(drop=True)

dict_sr = {'No Discriminación':'Non-discrimination', 'Bienestar Económico':'Economic Well-being',
           'Educación':'Education', 'Trabajo':'Working Conditions', 'Alimentación':'Nutrition',
           'Salud':'Health', 'Vivienda':'Housing', 'Medio Ambiente Sano' : 'Healthy Environment',
           'Seguridad Social': 'Social Security'}

for i,row in df22_sr.iterrows():
    df22_sr.loc[i,"Social_Right"] = dict_sr[df22_sr.loc[i,"Social_Right"]]

dict_col = {'Non-discrimination':'#CC89F1',
                'Economic Well-being':'#F8DF62',
                'Education':'#C65911',
                'Working Conditions':'#FFC000',
                'Nutrition':'#9EA7BC',
                'Health':'#A9D08E',
                'Housing':'#F99D5F',
                'Healthy Environment':'#9FEBD2',
                'Social Security':'#8EA9DB'
    }

SR = [i for i in dict_col]

dict_label = {'Non-discrimination':'Non\nDiscrimination',
                'Economic Well-being':'Economic\nWell-being',
                'Education':'Education',
                'Working Conditions':'Working\nConditions',
                'Nutrition':'Nutrition',
                'Health':'Health',
                'Housing':'Housing',
                'Healthy Environment':'Healthy\nEnvironment',
                'Social Security':'Social\nSecurity'
    }





i=0
plt.figure(figsize=(8,4.5))
for sr in SR:
    plt.bar(i+1, round((df22_sr.loc[df22_sr.Social_Right==sr,"SR_Perc"])*100,2), color=dict_col[sr])
    plt.text(i+1, (df22_sr.loc[df22_sr.Social_Right==sr,"SR_Perc"])*100 + 0.2, str(round(df22_sr.loc[df22_sr.Social_Right==sr,"SR_Perc"].values[0]*100,2)), 
             ha="center", va="center", fontsize=12,
             bbox=dict(facecolor='none', edgecolor='none', boxstyle='round', pad=.1, linewidth=.1))
    i+=1
plt.ylim(0, 4)
plt.gca().set_xticks(range(len(SR)+1))
plt.ylabel('Average budget % in 2022', fontsize=14)
plt.xlabel('Social Rights', fontsize=14)
plt.gca().set_xticklabels([""] +[dict_label[i] for i in SR], fontsize=10, rotation=45)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(home+'figures/figure_A2.pdf')
plt.show()








df22_small["SR_Prop"] = df22_small.groupby("Social_Right")["ID_RMC"].transform("count")
df22_small["SR_Prop"] = df22_small["SR_Prop"] / len(df22_small)
df22_sr_prop = df22_small[["Social_Right", "SR_Prop"]].copy().drop_duplicates().reset_index(drop=True)
for i,row in df22_sr_prop.iterrows():
    df22_sr_prop.loc[i,"Social_Right"] = dict_sr[df22_sr_prop.loc[i,"Social_Right"]]






i=0
plt.figure(figsize=(8,4.5))
for sr in SR:
    plt.bar(i+1, round((df22_sr_prop.loc[df22_sr_prop.Social_Right==sr,"SR_Prop"])*100,2), color=dict_col[sr])
    plt.text(i+1, (df22_sr_prop.loc[df22_sr_prop.Social_Right==sr,"SR_Prop"])*100 + 2, str(round(df22_sr_prop.loc[df22_sr_prop.Social_Right==sr,"SR_Prop"].values[0]*100,2)), 
             ha="center", va="center", fontsize=12,
             bbox=dict(facecolor='none', edgecolor='none', boxstyle='round', pad=.1, linewidth=.1))
    i+=1
plt.ylim(-2, 40)
plt.gca().set_xticks(range(len(SR)+1))
plt.ylabel('Programs distribution in 2022', fontsize=14)
plt.xlabel('Social Rights', fontsize=14)
plt.ylim(0, 40)
plt.gca().set_xticklabels([""] +[dict_label[i] for i in SR], fontsize=10, rotation=45)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(home+'figures/figure_A1.pdf')
plt.show()










