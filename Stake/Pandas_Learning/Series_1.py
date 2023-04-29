import pandas as pd
import numpy as np
from datetime import date, time, datetime, timedelta
s_ndarry = pd.Series({'2019-01-11' : 0., '2019-01-11' : 1.}, index = ['2019-01-11', '2019-01-12', '2019-01-13', '2019-01-14'])
print(f'datetime.resolution: {datetime.resolution}')
datetime_obj = datetime(2016, 10, 26, 10, 23, 15, 1)
print(f'strftime() : {datetime_obj.strftime("%m%d")}')