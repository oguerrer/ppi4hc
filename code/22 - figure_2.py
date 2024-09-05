import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


home =  os.getcwd()[:-4]

#########################
#########################
#
# Creates figure 2
#
#########################
#########################

paises = sorted(['United States', 'Mexico', 'Chile', 'Peru', 'Colombia', 'Brazil', 'Argentina', 'Uruguay',
          'Poland', 'Republic of Korea', 'Malaysia', 'Turkey', 'Thailand', 'Romania'])


df = pd.read_excel(home+'data/raw/penn_tables.xlsx', sheet_name=2)
df = df[['country', 'year', 'hc', 'ctfp', 'cn', 'emp', 'avh', 'pop', 'cgdpe']]
df = df[df['country'].isin(paises)]
df = df[(df['year'] >= 1989) & (df['year'] <= 2019)]
df = df.sort_values(['country', 'year'])
df['clr'] = df['cn'] / df['emp']
df['cor'] = df['cn'] / df['cgdpe']
df['gdppc'] = df['cgdpe'] / df['emp']


def growth_rate(series):
    return ((series.pct_change() + 1).prod() ** (1/(len(series)-1)) - 1)*100

growth_rates = df.groupby('country').agg({
    'emp': growth_rate,
    'ctfp': growth_rate,
    'hc': growth_rate,
    'clr': growth_rate,
    'cor': growth_rate,
    'gdppc': growth_rate
})

growth_rates.columns = ['li', 'agtfp', 'aghc', 'agclr', 'cdepp', 'aggdppc']



plt.figure(figsize=(8, 4.5))
bar_width = 0.15
index = np.arange(len(paises))
plt.bar(index, growth_rates['li'], bar_width, label='labor intensity', color='b')
plt.bar(index + bar_width, growth_rates['aghc'], bar_width, label='human capital', color='g')
plt.bar(index + 2*bar_width, growth_rates['cdepp'], bar_width, label='capital deepening', color='r')
plt.bar(index + 3*bar_width, growth_rates['agtfp'], bar_width, label='total factor productivity', color='c')
# plt.bar(index + 4*bar_width, growth_rates['aggdppc'], bar_width, label='GDP per capita growth', color='m')
plt.scatter(index + 1.5*bar_width, growth_rates['aggdppc'], color='k', s=25, label='GDP per capita')
plt.ylim(-2.5, 5)
plt.xlim(-.25, 13.75)
plt.ylabel('growth rates (%)', fontsize = 14)
# plt.title('Growth Rates by Country and Factor', fontsize = 16)
plt.xticks(index + 2*bar_width, paises, rotation=45, ha='right')
plt.legend(fontsize=8, loc=8, ncol=3)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.tight_layout()
plt.savefig(home+'figures/figure_2.pdf')
plt.show()














