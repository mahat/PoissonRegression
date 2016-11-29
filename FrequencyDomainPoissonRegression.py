'''
Author: mahat
'''

import pandas as pd
import statsmodels.api as sm
from math import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt

# reading data
data = pd.read_csv('./data/VisitData.txt', delim_whitespace=True, header=0)

print '----- data head -----'
print data.head()

print '----- data description -----'
print data.describe()

# plotting hour vs count data
plt.plot(data['hour'], data['count'], 'o')
plt.title('Visitor counts vs hour')
plt.xlabel('Hour')
plt.ylabel('Visitor Count')
plt.show()

# in data hours are incremented in everyday
# however, time series data have generally cycles so we need to transform hour column into hour in a day
data['hourofday'] = data['hour'].apply(lambda x: ((x - 1) % 24) + 1)

print data.head()
# plotting histogram which is grouped by according to hourofday
totalVisit = np.zeros(24)
for index, row in data.iterrows():
    totalVisit[row['hourofday'] - 1] = totalVisit[row['hourofday'] - 1] + row['count']

width = 1 / 1.5
plt.bar(range(1, 25), totalVisit, width, color="blue")
plt.ylabel('Mean Count of Visitors')
plt.xlabel('Hour of the day')
plt.title('Histogram')
plt.show()

# from the bar chart, it can be said that there is a cycle
# we can convert hour data into frequency domain therefore we can handle cycles
data['w'] = data['hour'].apply(lambda h: (float(h) / 24) * 2 * pi)
# conversion to frequency domain
data['fdomain'] = data['w'].apply(lambda w: sin(w) + cos(w) + sin(2*w) + cos(2*w))
# applying poisson regression
# X
feat_cols = ['fdomain']
X = [elem for elem in data[feat_cols].values]

# adding costant to adding bias
X = sm.add_constant(X, prepend=False)
# Y
Y = [elem for elem in data['count'].values]
# building the model
poisson_mod = sm.Poisson(Y, X)
poisson_res = poisson_mod.fit(method="newton")
print(poisson_res.summary())

# predicted vals
predVals = poisson_res.predict(X)

# plotting results
plt.plot(data['hourofday'], data['count'], 'bo',label= 'visitor counts')
plt.hold(True)
plt.plot(data['hourofday'], predVals, 'ro-',label= 'predicted visitor counts')
plt.legend()
plt.title('Comparesion between distribution of count and predicted values')
plt.xlabel('Hour of day')
plt.ylabel('Visitor Count')
plt.hold(False)

plt.show()