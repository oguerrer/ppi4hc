import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, warnings

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

warnings.simplefilter("ignore")
sparsebn = importr('sparsebn')

home =  os.getcwd()[:-4]
os.chdir(home)


##################################################
##################################################
#
# Estimates network of interdependencies
# It requires the R library sparsebn and the
# Python binders Rpy2 to R
#
##################################################
##################################################


# ## Installing R packages (optional)
# import rpy2.robjects.packages as rpackages
# utils = rpackages.importr('utils')
# utils.chooseCRANmirror(ind=1)
# packnames = ('sparsebnUtils')
# from rpy2.robjects.vectors import StrVector
# names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
# if len(names_to_install) > 0:
#     utils.install_packages(StrVector(names_to_install))


df = pd.read_csv(home+'/data/clean/Outcomes/coneval_perf_corrected.csv', encoding='utf-8-sig')
colYears = [col for col in df.columns if col.isnumeric()]





S = (df[colYears].values[:,1::] - df[colYears].values[:,0:-1]).T

nr, nc = S.shape
Sr = rpy2.robjects.r.matrix(S, nrow=nr, ncol=nc)
rpy2.robjects.r.assign("S", Sr)

rpy2.robjects.r('''
    library(sparsebnUtils)
    library(sparsebn)
    library(ccdrAlgorithm)
    data <- sparsebnData(S, type = "continuous")
    dags.estimate <- sparsebn::estimate.dag(data)
    dags.param <- estimate.parameters(dags.estimate, data=data)
    selected.lambda <- select.parameter(dags.estimate, data=data)
    dags.final.net <- dags.estimate[[selected.lambda]]
    dags.final.param <- dags.param[[selected.lambda]]
    adjMatrix <- as(dags.final.param$coefs, "matrix")
    ''')
    
A = rpy2.robjects.globalenv['adjMatrix']



## correct the weights by removing potential false positives with extreme values
A[A<0] = 0
A[np.abs(A)>1] = 0


    

edges = []
for i, rowi in df.iterrows():
    for j, rowj in df.iterrows():
        if A[i,j] != 0:
            edges.append((i, j, A[i,j]))

dff = pd.DataFrame(edges, columns=['From', 'To', 'Weight'])   
dff.to_csv(home+'/data/clean/network.csv', index=False)


 
 

 



























































