'''
Author: mahat
'''
'''
Author: mahat
'''

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

#read data
data = pd.read_csv('./data/StudentData.csv', delimiter=',',header=0)

print '----- data head -----'
print data.head()
print '----- data description -----'
print data.describe()
histData = []
uniqProgs = sorted(data['prog'].unique())
for elem in uniqProgs:
    histData.append(data[data['prog'] == elem]['num_awards'].values)

# plotting histogram in order to see
plt.hist(tuple(histData),bins=10, normed=True,histtype='bar',label= map(lambda x: 'Prog '+ str(x),uniqProgs))
plt.legend()
plt.ylabel('Count')
plt.title('Histogram for each program')
plt.show()

# adding dummy variables in order to handle categorical data in prog
prog_dummies = pd.get_dummies(data['prog']).rename(columns=lambda x: 'prog_' + str(x))
dataWithDummies = pd.concat([data, prog_dummies], axis=1)
dataWithDummies .drop(['prog', 'prog_3'], inplace=True, axis=1)
dataWithDummies = dataWithDummies .applymap(np.int)

print dataWithDummies.head()

# applying poisson regression on data
# assuming variables are independent to each other
feat_cols = ['math', 'prog_1', 'prog_2']
X = [elem for elem in dataWithDummies[feat_cols].values]
# adding constant to adding bias
X = sm.add_constant(X, prepend=False)
Y = [elem for elem in dataWithDummies['num_awards'].values]

# building the model
poisson_mod = sm.Poisson(Y, X)
poisson_res = poisson_mod.fit(method="newton")
print(poisson_res.summary())


# testing the model
predVals = poisson_res.predict(X)

plt.plot(range(len(Y)), Y, 'r*-', range(len(Y)), predVals, 'bo-')
plt.title('Train dataset Real vs. Predicted Values')
plt.legend(['Real Values', 'Predicted Values'])
plt.show()