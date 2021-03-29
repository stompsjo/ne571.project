import wget

from datetime import date, timedelta

start_date = date(2019, 1, 1)
end_date = date(2019, 12, 31)
delta = timedelta(days=1)
while start_date <= end_date:
    url = 'https://docs.misoenergy.org/marketreports/' + start_date.strftime('%Y%m%d') + '_rt_lmp_final.csv'
    wget.download(url, 'data/lmp/' + start_date.strftime('%Y%m%d') + '.csv')
    start_date += delta
