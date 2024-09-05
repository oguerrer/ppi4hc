import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
warnings.simplefilter("ignore")
import matplotlib.patches as mpatches


home =  os.getcwd()[:-4]

#########################
#########################
#
# Creates figure 9
#
#########################
#########################


df_id = pd.read_excel(home + '/data/clean/Outcomes/coneval_labels.xlsx', sheet_name="Sheet1")
df_id.sort_values(by=['Derecho Social o Bienestar Económico (directo)'], inplace=True, ignore_index=True)

df_trim = df_id[['Label', 'Derecho Social o Bienestar Económico (directo)', 'Colour',
                'Stage Of Life_1', 'Stage Of Life_2', 'Stage Of Life_3', 
                'Stage Of Life_4', 'Stage Of Life_5']].copy()

df_trim.set_index("Label", drop=True, inplace=True)

dict_id = df_trim.to_dict(orient="index")

for i in dict_id:
    dict_id[i]["Stage of Life"] = [dict_id[i]["Stage Of Life_1"], dict_id[i]["Stage Of Life_2"],
                                       dict_id[i]["Stage Of Life_3"], dict_id[i]["Stage Of Life_4"],
                                       dict_id[i]["Stage Of Life_5"]]
    
    
df_patches = df_id[["Derecho Social o Bienestar Económico (directo)", "Colour" ]].copy().drop_duplicates().reset_index(drop=True)
list_patches = [mpatches.Patch(color=df_patches.loc[i,"Colour"], label=df_patches.loc[i,"Derecho Social o Bienestar Económico (directo)"]) for i in range(len(df_patches))]


df1 = df_id['Stage Of Life_1'].value_counts(dropna=True)
df2 = df_id['Stage Of Life_2'].value_counts(dropna=True)
df3 = df_id['Stage Of Life_3'].value_counts(dropna=True)
df4 = df_id['Stage Of Life_4'].value_counts(dropna=True)
df5 = df_id['Stage Of Life_5'].value_counts(dropna=True)

df_count = pd.DataFrame({'SoL1': df1, 'SoL2': df2, 'SoL3': df3, 'SoL4': df4, 'SoL5': df5})
df_count.loc[:,"Sum"] = df_count.sum(axis=1)


fig = plt.figure(figsize=(8,4.5))
child1 = 0
youth1 = 0
youth2 = 0
yadult1 = 0
yadult2 = 0
adult1 = 0
adult2 = 0
elder1 = 0
columny = 0
columnya = 0
columna = 0

for i in dict_id:
       
    if "Elderly" in dict_id[i]["Stage of Life"]:
        plt.text(2.6, elder1, i, color='w', 
          horizontalalignment='center', fontsize=7, weight='bold',
          bbox=dict(facecolor=dict_id[i]["Colour"], pad=.2, edgecolor='w', boxstyle='round'))
        elder1 += 1
    if "Middle-age Adults" in dict_id[i]["Stage of Life"]:
        if columna == 0 :
            plt.text(2.2, adult1, i, color='w', 
              horizontalalignment='center', fontsize=7, weight='bold',
              bbox=dict(facecolor=dict_id[i]["Colour"], pad=.2, edgecolor='w', boxstyle='round'))
            adult1 += 1
            columna += 1
        elif columna == 1 :
            plt.text(1.9, adult2, i, color='w', 
                  horizontalalignment='center', fontsize=7, weight='bold',
                  bbox=dict(facecolor=dict_id[i]["Colour"], pad=.2, edgecolor='w', boxstyle='round'))
            adult2 += 1
            columna = 0
    if "Young Adults" in dict_id[i]["Stage of Life"]:
        if columnya==0:
            plt.text(1.5, yadult1, i, color='w', 
                  horizontalalignment='center', fontsize=7, weight='bold',
                  bbox=dict(facecolor=dict_id[i]["Colour"], pad=.2, edgecolor='w', boxstyle='round'))
            yadult1 += 1
            columnya += 1
        elif columnya==1:
            plt.text(1.2, yadult2, i, color='w', 
                  horizontalalignment='center', fontsize=7, weight='bold',
                  bbox=dict(facecolor=dict_id[i]["Colour"], pad=.2, edgecolor='w', boxstyle='round'))
            yadult2 += 1
            columnya = 0
    if "Youths" in dict_id[i]["Stage of Life"]:
        if columny==0:
            plt.text(.8, youth1, i, color='w', 
                  horizontalalignment='center', fontsize=7, weight='bold',
                  bbox=dict(facecolor=dict_id[i]["Colour"], pad=.2, edgecolor='w', boxstyle='round'))
            youth1 += 1
            columny += 1
        elif columny==1:
            plt.text(.5, youth2, i, color='w', 
                  horizontalalignment='center', fontsize=7, weight='bold',
                  bbox=dict(facecolor=dict_id[i]["Colour"], pad=.2, edgecolor='w', boxstyle='round'))
            youth2 += 1
            columny = 0
    if "Children" in dict_id[i]["Stage of Life"]:
        plt.text(.1, child1, i, color='w', 
              horizontalalignment='center', fontsize=7, weight='bold',
              bbox=dict(facecolor=dict_id[i]["Colour"], pad=.2, edgecolor='w', boxstyle='round'))
        child1 += 1
                
plt.text(.1, -max([child1, youth1, youth2, yadult1, yadult2, adult1, adult2, elder1])*.1, ' CHILDREN ', color='w', 
          horizontalalignment='center', fontsize=8, weight='bold',
          bbox=dict(facecolor='#FEC896', edgecolor='w', boxstyle='round'))
plt.text(.65, -max([child1, youth1, youth2, yadult1, yadult2, adult1, adult2, elder1])*.1, '               YOUTHS               ', color='w', 
          horizontalalignment='center', fontsize=8, weight='bold',
          bbox=dict(facecolor='#FDA453', edgecolor='w', boxstyle='round'))
plt.text(1.35, -max([child1, youth1, youth2, yadult1, yadult2, adult1, adult2, elder1])*.1, '          YOUNG ADULTS          ', color='w', 
          horizontalalignment='center', fontsize=8, weight='bold',
          bbox=dict(facecolor='#FD8619', edgecolor='w', boxstyle='round'))
plt.text(2.05, -max([child1, youth1, youth2, yadult1, yadult2, adult1, adult2, elder1])*.1, '         MIDDLE-AGE ADULTS         ', color='w', 
          horizontalalignment='center', fontsize=8, weight='bold',
          bbox=dict(facecolor='#D86802', edgecolor='w', boxstyle='round'))
plt.text(2.6, -max([child1, youth1, youth2, yadult1, yadult2, adult1, adult2, elder1])*.1, ' ELDERLY ', color='w', 
          horizontalalignment='center', fontsize=8, weight='bold',
          bbox=dict(facecolor='#AC5302', edgecolor='w', boxstyle='round'))

plt.ylim(-max([child1, youth1, youth2, yadult1, yadult2, adult1, adult2, elder1])*.15, (max([child1, youth1, youth2, yadult1, yadult2, adult1, adult2, elder1])+1)*1)
plt.xlim(-.1, 2.8)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.xlabel('age group', fontsize=14)
plt.yticks([])
plt.xticks([])
plt.legend(handles=list_patches, fontsize=9, bbox_to_anchor=(0.38, .96))
plt.tight_layout()
plt.savefig(home+'figures/figure_9.pdf', bbox_inches="tight")
plt.show()

    













