# Francesco Turci
# Reproducing the Welch t-test statistics of scipy
import numpy as np
from scipy import stats
import scipy

#Let's suppose we have a binary problem: two populations A, B need choose between options (0,1). Are there any statistically meaningful differences between the two?

# samples: they do not neet to have equal size
nza,noa = 20,30
nzb,nob = 10, 60
A = np.concatenate((np.zeros(nza), np.ones(noa)))
B = np.concatenate((np.zeros(nzb), np.ones(nob)))
ttest = stats.ttest_ind(A,B,equal_var=False) #key: equal_var needs to be False for the Welch test t-test
print("Scipy result")
print(ttest)

import pandas as pd

def my_welch_ttest(A,B,alpha=0.05):
    ma = A.mean()
    mb = B.mean()
    na,nb = len(A),len(B)
    stea = A.std(ddof=1)/np.sqrt(na)
    steb = B.std(ddof=1)/np.sqrt(nb)
    sed = np.sqrt(stea**2+steb**2)
    diff = ma-mb
    t_stat = diff/sed
    # degrees of freedom
    # naively it could be df = na + nb - 2
    # but truly it is https://en.wikipedia.org/wiki/Welch%27s_t-test#Calculations
    df = ((stea**2+steb**2)**2)/(stea**4/(na-1)+steb**4/(nb-1))
    print(df)
    critical_value = stats.t.ppf(1.0 - alpha, df)
    # p_value = (1 - stats.t.cdf(abs(t_stat), df)) * 2 #2-sided
    p_value = 2*scipy.special.stdtr(83.18487249884447, -np.abs(t_stat))
    return t_stat,p_value

print("MY result:")

print(my_welch_ttest(A,B))
