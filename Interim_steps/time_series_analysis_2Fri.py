# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 13:17:54 2021

@author: CS_Knit_tinK_SC
"""

import numpy as np
import pandas as pd
from pathlib import Path
%matplotlib inline

#%%

# Return Forecasting: Read Historical Daily Yen Futures Data¶

# In this notebook, you will load historical Dollar-Yen exchange rate futures data 
# and apply time series analysis and modeling to determine whether there is any predictable behavior.

#%%

# Futures contract on the Yen-dollar exchange rate:
# This is the continuous chain of the futures contracts that are 1 month to expiration
# Futures contract on the Yen-dollar exchange rate:
# This is the continuous chain of the futures contracts that are 1 month to expiration

# absolute path method  - while in process
# C:\Users\CS_Knit_tinK_SC\Documents\GitHub\HW_7_TS_ML_Inputs-U10
filepath="C:/Users/CS_Knit_tinK_SC/Documents/GitHub/HW_7_TS_ML_Inputs-U10/Resources/yen.csv"
yen_futures = pd.read_csv(filepath, index_col="Date", infer_datetime_format=True, parse_dates=True)


# relative path method - as of submittal 
#yen_futures = pd.read_csv(
#    Path('../Resources/yen.csv'), index_col="Date", infer_datetime_format=True, parse_dates=True
#)
yen_futures.head()

#%%

# Trim the dataset to begin on January 1st, 1990
yen_futures = yen_futures.loc["1990-01-01":, :]
print(yen_futures.head())

#%%

# Return Forecasting: Initial Time-Series Plotting¶

# Start by plotting the "Settle" price. Do you see any patterns, long-term and/or short?

#%%

# Plot just the "Settle" column from the dataframe using a line plot:
# title to add: yen Futures Settle Prices
# df.Close.plot()
yen_futures['Last'].plot(title='Yen Futures Settle Prices')

#%%

# Decomposition Using a Hodrick-Prescott Filter

# Using a Hodrick-Prescott Filter, decompose the Settle price into a trend and noise.

#%%

import statsmodels.api as sm

# Apply the Hodrick-Prescott Filter by decomposing the "Settle" price into two separate series:
# Apply the Augmented Dickey-Fuller test to determine if the above is stationary

from statsmodels.tsa.stattools import adfuller

# Store the results of the test in the variable result
result = adfuller(yen_futures.Settle)

# Access the contents of the results:
print('ADF Statistic: %f' % result[0])
print('p-value:        %f ' % result[1])
print('Lags used:      %d' % result[2])
print('Critical Values:')
for key, value in result[4].items():
    print((key, value))
#%%
