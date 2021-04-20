import numpy as np
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=[5,4])
'''
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
'''
plt.rcParams['font.size'] = 12
data = np.genfromtxt('data/clearing-price/2019clearing.csv', delimiter=',')
#print(data[0,1:].reshape((len(data[0,1:]),1)))
#print(data.shape)


timeseries = np.array([])
for row in data[:-1]:
    #print(row[0])
    timeseries = np.append(timeseries, row[1:])

demand = np.genfromtxt('MIWI-demand-MWh.csv', delimiter=',')
print(demand.shape)
print(timeseries.shape)
#demand = demand[:,-1]
ave_price = np.array([])
ave_demand = np.array([])
for i in range(len(timeseries)):
    if i == 0:
        ave_price = np.append(ave_price, timeseries[0])
        ave_demand = np.append(ave_demand, demand[0][-1])
    else:
        avep = np.mean(timeseries[max(i-7,0):i][-1])
        ave_price = np.append(ave_price, avep)

        aved = np.mean(demand[max(i-7,0):i])
        ave_demand = np.append(ave_demand, aved)
'''
# calc the trendline
z = np.polyfit(ave_demand, ave_price, 1)
p = np.poly1d(z)
plt.plot(ave_demand,p(ave_price),"r--")
# the line equation:
print("y=%.6fx+(%.6f)"%(z[0],z[1]))
plt.scatter(ave_demand, ave_price)
plt.grid()
plt.xlabel('Demand [MWh]')
plt.ylabel('Clearing Price [$/MW]')
plt.title('7-day Rolling Average Pseudo-Michigan Hub Demand Curve')
plt.savefig('rolling-ave-demand-curve.png')
'''
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
'''
###### Clearing Price Duration Curve for Michigan Hub
X, F = np.unique(list(reversed(sorted(timeseries))), return_index=True)
#F = F/float(F.max())
#ax = plt.gca()
#ax.invert_xaxis()
#print(list(reversed(sorted(demand[:,-1]))))
plt.plot(F,X)
plt.grid()
plt.xlim(0,100)
plt.ylim(0)
plt.xlabel('Cumulative Hours at Clearing Price')
plt.ylabel('Clearing Price [$/MW]')
#plt.title('WI-MI Clearing Price Duration Curve')
plt.savefig('clearing_duration_curve.png')
'''
'''
lmp = np.genfromtxt('data/lmp/lmp.csv', delimiter=',')
nodelmp = np.array([])
for row in lmp[:-1]:
    #print(row[0])
    nodelmp = np.append(nodelmp, row[4:])

print(nodelmp.shape)
X, F = np.unique(list(reversed(sorted(nodelmp))), return_index=True)
ax.plot(F,X)
plt.grid()
#plt.xlim(0,max(F))
ax.set_xlim(0,max(F))
ax.set_ylim(0)
#plt.ylim(0)

#ax.set_xlabel('Cumulative Hours at Clearing Price')
#ax.set_ylabel('Clearing Price [$/MW]')
#ax.set_title('Palisades Node Clearing Price Duration Curve')
plt.tight_layout()
plt.savefig('node_clearing_duration_curve.png')
'''
'''
lmp = np.genfromtxt('data/lmp/lmp.csv', delimiter=',')
nodelmp = np.array([])
for row in lmp[:-1]:
    #print(row[0])
    nodelmp = np.append(nodelmp, row[4:])

mean = np.mean(nodelmp)
standard_deviation = np.std(nodelmp)
distance_from_mean = abs(nodelmp - mean)
max_deviations = 2
not_outlier = distance_from_mean < max_deviations * standard_deviation
no_outliers = nodelmp[not_outlier]
mask = np.isin(nodelmp, no_outliers)
xs = demand[:,-1][mask]

# calc the trendline
z = np.polyfit(xs, no_outliers, 1)
p = np.poly1d(z)
plt.plot(xs,p(xs),"r--")
# the line equation:
print('Mean:',mean)
print('STD:',standard_deviation)
print("y=%.6fx+(%.6f)"%(z[0],z[1]))
print(nodelmp.shape)
plt.scatter(demand[:,-1],nodelmp,s=2)
plt.ylim(0,200)
#plt.hlines(mean,10000,33000,color='black')
plt.xlim(10000,33000)
plt.grid()
plt.xlabel('Demand [MWh]')
plt.ylabel('Clearing Price [$/MW]')
plt.title('Palisades Node Demand Curve')
plt.savefig('fitted-node-demand-curve.png')
'''

mean = np.mean(timeseries)
standard_deviation = np.std(timeseries)
distance_from_mean = abs(timeseries - mean)
max_deviations = 2
not_outlier = distance_from_mean < max_deviations * standard_deviation
no_outliers = timeseries[not_outlier]
mask = np.isin(timeseries, no_outliers)
xs = demand[:,-1][mask]
z = np.polyfit(xs, no_outliers, 1)
plt.scatter(demand[:,-1], timeseries,s=2)
p = np.poly1d(z)
plt.plot(xs,p(xs),"r--")
print('Mean:',mean)
print('STD:',standard_deviation)
print("y=%.6fx+(%.6f)"%(z[0],z[1]))
plt.grid()
plt.ylim(0,100)
plt.hlines(mean,10000,35000,colors='black')
plt.xlim(10000,35000)
plt.xlabel('Demand [MWh]')
plt.ylabel('Clearing Price [$/MW]')
plt.title('Pseudo-Michigan Hub Demand Curve')
plt.savefig('mean-fitted-demand-curve.png')

'''
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import datetime
numh = 8736
base = datetime.datetime(2019,1,1)
date_list = [base - datetime.timedelta(hours=x) for x in range(0, numh)]
plt.plot(date_list,timeseries)
# Set the locator
locator = mdates.MonthLocator()  # every month
# Specify the format - %b gives us Jan, Feb...
fmt = mdates.DateFormatter('%b')
X = plt.gca().xaxis
X.set_major_locator(locator)
# Specify formatter
X.set_major_formatter(fmt)
plt.grid()
plt.xlabel('Months of 2019')
plt.ylabel('$/MW')
plt.title('Michigan Hub Clearing Prices')

plt.savefig('clearing-results.png')
'''
