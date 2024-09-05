import matplotlib.pyplot as plt
import os
import pandas as pd


home =  os.getcwd()[:-4]

#########################
#########################
#
# Creates figure 1
#
#########################
#########################


paises = ['United States', 'Mexico', 'Chile', 'Peru', 'Colombia', 'Brazil', 'Argentina', 'Uruguay', 
          'Poland', 'Republic of Korea', 'Malaysia', 'Turkey', 'Thailand', 'Romania']
df = pd.read_excel(home+'/data/raw/penn_tables.xlsx', sheet_name=2)


df = df[['country', 'rgdpe', 'pop', 'year']]
df = df[df['country'].isin(paises)]
df_pivot = df.pivot(index='country', columns='year', values=['rgdpe', 'pop'])
df_rgdppc = df_pivot['rgdpe'] / df_pivot['pop']
df_gapusa = df_rgdppc.div(df_rgdppc.loc['United States'])
df_gapusa = df_gapusa.T



fig, ax = plt.subplots(figsize=(8, 4.5))
ax.set_ylabel('Ratio of GDP per capita\n(country / US)', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
plt.xticks(df_gapusa.index[::2], rotation=45)  # Mostrar cada cuatro años para evitar superposición
filtered_df_gapusa = df_gapusa.loc['1982':'2019']
for pais in df_gapusa.columns:
    if pais != 'United States':
        if pais == 'Mexico':
            ax.plot(filtered_df_gapusa.index, filtered_df_gapusa[pais], '-.k', 
                    label=pais, linewidth=3)
        else:
            ax.plot(filtered_df_gapusa.index, filtered_df_gapusa[pais], label=pais, linewidth=1.5, zorder=-1)
plt.ylim(.07, .85)
plt.xlim(1982, 2018)
ax.legend(fontsize=8, loc='upper center', ncol=5)
specific_point = filtered_df_gapusa.index[3]  
specific_value = filtered_df_gapusa.loc[specific_point, 'Mexico']
plt.gca().spines[['right', 'top']].set_visible(False)
plt.savefig(home+'figures/figure_1.pdf')
plt.show()








































