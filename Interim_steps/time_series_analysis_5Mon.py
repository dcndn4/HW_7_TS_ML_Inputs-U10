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
yen_futures['Settle'].plot(title='Yen Futures Settle Prices')

#%%

# Decomposition Using a Hodrick-Prescott Filter

# Using a Hodrick-Prescott Filter, decompose the Settle price into a trend and noise.



#%%

import statsmodels.api as sm

# Apply the Hodrick-Prescott Filter by decomposing the "Settle" price into two separate series:


ts_noise, ts_trend = sm.tsa.filters.hpfilter(yen_futures['Settle'])

#%%
data_result = yen_futures[['Settle']]

#%%
# data_result.join(ts_trend)

ts_new = pd.concat([ts_trend, ts_noise], axis=1)


#%%

all_three=pd.concat([data_result, ts_new], axis=1)

all_three.rename(columns={'Settle_trend':'Trend','Settle_cycle':'Noise'}, inplace=True)

#%%

# Plot the Settle Price vs. the Trend for 2015 to the present
settle_trend_2015 = all_three.loc['2015':'2020']

#%%
settle_trend_2015.tail()

#%%

settle_trend_2015[["Settle", "Trend"]].plot(title="Settle Price vs. Trend - 2015-2020")

#%%

# Plot the Settle Noise
settle_trend_2015[["Noise"]].plot(title="Noise - 2015-2020")

#%%

# Forecasting Returns using an ARMA Model

# Using futures Settle Returns, estimate an ARMA model

#     ARMA: Create an ARMA model and fit it to the returns data. Note: Set the AR and MA ("p" and "q") parameters to p=2 and q=1: order=(2, 1).
#     Output the ARMA summary table and take note of the p-values of the lags. Based on the p-values, is the model a good fit (p < 0.05)?
#     Plot the 5-day forecast of the forecasted returns (the results forecast from ARMA model)


#%%

# Create a series using "Settle" price percentage returns, drop any nan"s, and check the results:
# (Make sure to multiply the pct_change() results by 100)
# In this case, you may have to replace inf, -inf values with np.nan"s
returns = (yen_futures[["Settle"]].pct_change() * 100)
returns.rename(columns={'Settle':'Return_Amt'}, inplace=True)
returns = returns.replace(-np.inf, np.nan).dropna()
returns.tail()

#%%

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA

# Estimate and ARMA model using statsmodels (use order=(2, 1))
model = ARMA(returns.values, order=(2, 1))  # (AR = past values and errors, MA = past errors)

# Fit the model and assign it to a variable called results
results = model.fit()

#%%

# Output model summary results:
print(results.summary())

#%%

# Plot the 5 Day Returns Forecast
pd.DataFrame(results.forecast(steps=5)[0]).plot(title="5 Day Returns Forecast")

#%%

# Forecasting the Settle Price using an ARIMA Model

#    Using the raw Yen Settle Price, estimate an ARIMA model.
#        Set P=5, D=1, and Q=1 in the model (e.g., ARIMA(df, order=(5,1,1))
#        P= # of Auto-Regressive Lags, D= # of Differences (this is usually =1), Q= # of Moving Average Lags
#    Output the ARIMA summary table and take note of the p-values of the lags. Based on the p-values, is the model a good fit (p < 0.05)?
#    Construct a 5 day forecast for the Settle Price. What does the model forecast will happen to the Japanese Yen in the near term?


#%%

from statsmodels.tsa.arima_model import ARIMA

# Estimate and ARIMA Model:
# Hint: ARIMA(df, order=(p, d, q))
model = ARIMA(returns['Settle'], order=(5, 1, 1))

# Fit the model
results = model.fit()
#%%

# Output model summary results:
print(results.summary())

#%%

# Plot the 5 Day Price Forecast
pd.DataFrame(results.forecast(steps=5)[0]).plot(title="5 Day Futures Price Forecast")

#%%

# Volatility Forecasting with GARCH

# Rather than predicting returns, let's forecast near-term volatility of 
# Japanese Yen futures returns. 
# Being able to accurately predict volatility 
# will be extremely useful if we want to trade in derivatives or quantify our maximum loss.

# Using futures Settle Returns, estimate an GARCH model

# GARCH: Create an GARCH model and fit it to the returns data. Note: Set the parameters to p=2 and q=1: order=(2, 1).
#  Output the GARCH summary table and take note of the p-values of the lags. Based on the p-values, is the model a good fit (p < 0.05)?
#   Plot the 5-day forecast of the volatility.



#%%

import arch
from arch import arch_model
#%%



# Estimate a GARCH model:
# 'p' and 'q'  are akin to the 'p' and 'q' of an ARMA model.
# 'vol="GARCH"' means that we're using a GARCH model.
# The 'mean="Zero"' means that we're estimating a GARCH.
model = arch_model(returns, mean="Zero", vol="GARCH", p=2, q=1)

# Fit the model
res = model.fit(disp="off")


#%%

# Summarize the model results
print(res.summary())

#%%

# Plot the model estimate of the annualized volatility
fig = res.plot(annualize='D')

#%%

# Find the last day of the dataset
#last_day = res.index.max().strftime('%Y-%m-%d')
#last_day
print(yen_futures.tail())

#%%

## Create a 5 day forecast of volatility
forecast_horizon = 5
# Take the last day of the data we used above. 
# Forecast horizon is 5, so resulting 'h.1', 'h.2', 'h.3', 'h.4', and 'h.5' 
# are the forecasts for the following 5 days.
forecasts = res.forecast(start='2019-10-15', horizon=forecast_horizon, reindex=True)
print(forecasts)

#%%

# Annualize the forecast
intermediate = np.sqrt(forecasts.variance.dropna() * 252)
print(intermediate)

#%%

# Transpose the forecast so that it is easier to plot
final = intermediate.dropna().T
print(final.head())

#%%

# Plot the final forecast
final.plot(title='GARCH volatility forecast')

#%%

Conclusions

Based on your time series analysis, would you buy the yen now?

Is the risk of the yen expected to increase or decrease?

Based on the model evaluation, would you feel confident in using these models for trading?

Type Markdown and LaTeX: α2

#%%