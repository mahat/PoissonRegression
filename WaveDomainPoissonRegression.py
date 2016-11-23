'''
Author: mahat
'''

import pandas as pd
import statsmodels.api as sm
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
plt.show()

# in data hours are incremented in everyday
# however, time series data have generally cycles so we need to transform hour column into hour in a day
data['hourofday'] = data['hour'].apply(lambda x: ((x - 1) % 24) + 1)

print data.head()
# plotting histogram which is grouped by according to hourofday
histData = []
hours = range(1, 25)
for elem in hours:
    histData.append(data[data['hourofday'] == elem]['count'].values)

plt.hist(tuple(histData), bins=2, normed=True, histtype='bar')
plt.show()