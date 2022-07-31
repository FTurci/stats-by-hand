# Francesco Turci
# implementing the one-way ANOVA test for the
# We measure heights from three subpopulations A,B,C and

import numpy as np
import pandas as pd
# simulate the measurement
na,nb,nc = 10,20,15
A = np.random.uniform(150,190, size=na)
B = np.random.normal(160,12, size=nb)
C = np.random.uniform(155,178, size=nc)

# compile a dataframe
df = pd.DataFrame()
df['population'] = (['A']*len(A)+['B']*len(B)+['C']*len(C))
df['heights'] = list(A)+list(B)+list(C)
print ("The averages are")
print(df.groupby('population').mean())
print("Are the three populations different?")
# using statsmodels

import statsmodels.api as sm
from statsmodels.formula.api import ols

linear_model = ols('heights ~ population',data=df).fit()
table = sm.stats.anova_lm(linear_model)
print(table)
