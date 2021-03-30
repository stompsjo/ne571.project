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

#demand = demand[:,-1]
ave_price = np.array([])
ave_demand = np.array([])
for i in range(len(timeseries)):
    ave = np.mean(timeseries[max(i-7,0):i])
    ave_price = np.append(ave_price, ave)

    ave = np.mean(demand[max(i-7,0):i])
    ave_demand = np.append(ave_demand, ave)
'''
plt.scatter(ave_demand, ave_price)
plt.grid()
plt.xlabel('Demand [MWh]')
plt.ylabel('Clearing Price [$/MW]')
plt.title('7-day Rolling Average Pseudo-Michigan Hub Demand Curve')
plt.savefig('rolling-ave-demand-curve.png')
'''
lmp = np.genfromtxt('data/lmp/lmp.csv', delimiter=',')
nodelmp = np.array([])
for row in lmp[:-1]:
    #print(row[0])
    nodelmp = np.append(nodelmp, row[4:])

X, F = np.unique(list(reversed(sorted(demand[:,-1]))), return_index=True)
#F = F/float(F.max())
#ax = plt.gca()
#ax.invert_xaxis()
#print(list(reversed(sorted(demand[:,-1]))))
plt.plot(F,X)
plt.grid()
plt.xlim(0,max(F))
plt.ylim(0)
plt.xlabel('Cumulative Hours at Load')
plt.ylabel('Hourly Demand [MW]')
plt.title('WI-MI Load Duration Curve')
plt.savefig('load-curve.png')

'''
print(nodelmp.shape)
plt.scatter(demand[:,-1],nodelmp)
plt.grid()
plt.xlabel('Demand [MWh]')
plt.ylabel('Clearing Price [$/MW]')
plt.title('Palisades Node Demand Curve')
plt.savefig('node-demand-curve.png')
'''
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
