import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data/clearing-price/2019clearing.csv', delimiter=',')
#print(data[0,1:].reshape((len(data[0,1:]),1)))
#print(data.shape)
fig, ax = plt.subplots()


timeseries = np.array([])
for row in data[:-1]:
    print(row[0])
    timeseries = np.append(timeseries, row[1:])

demand = np.genfromtxt('MIWI-demand-MWh.csv', delimiter=',')
print(demand[:,-1])
print(demand[:,-1].shape)
print(timeseries.shape)

lmp = np.genfromtxt('data/lmp/lmp.csv', delimiter=',')
nodelmp = np.array([])
for row in lmp[:-1]:
    #print(row[0])
    nodelmp = np.append(nodelmp, row[4:])

print(nodelmp.shape)
plt.scatter(demand[:,-1],nodelmp)
plt.savefig('node-demand-curve.png')
'''
plt.scatter(demand[:,-1], timeseries)
plt.grid()
plt.xlabel('Demand [MWh]')
plt.ylabel('Clearing Price [$/MW]')
plt.title('Pseudo-Michigan Hub Demand Curve')
plt.savefig('demand-curve.png')
'''
'''
print(timeseries)
print(timeseries.shape)
plt.plot(timeseries)
ax.set_xticklabels(data[:,0].astype(int).astype(str))
plt.grid()
plt.xlabel('Date')
plt.ylabel('$/MW')
plt.title('Michigan Hub Clearing Prices')

plt.savefig('clearing-results.png')
'''
