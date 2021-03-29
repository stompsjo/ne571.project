import pandas as pd
import numpy as np

#data = pd.read_csv('20190101.csv',delimiter=',',header=11,skipfooter=12)

from datetime import date, timedelta

start_date = date(2019, 1, 1)
end_date = date(2019, 12, 31)
delta = timedelta(days=1)
results = np.empty((0,25))
while start_date <= end_date:
    data = pd.read_csv('data/clearing-price/'+start_date.strftime('%Y%m%d')+'.csv',delimiter=',',header=11,skipfooter=12)
    demand = data['Michigan Hub'].to_numpy().reshape((1,len(data['Michigan Hub'])))
    results = np.append(results, np.append([start_date.strftime('%Y%m%d')], demand).reshape((1,25)),axis=0)
    print(results.shape)
    start_date += delta

print(results.shape)

pd.DataFrame(results).to_csv("2019clearing.csv")
