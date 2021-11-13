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

#C:\Users\CS_Knit_tinK_SC\Documents\GitHub\HW_7_TS_ML_Inputs-U10\Resources
#filepath="C:/Users/CS_Knit_tinK_SC/Documents/My Data Sources/110921/liquor_sales.csv"
#df = pd.read_csv(filepath)


filepath="C:/Users/CS_Knit_tinK_SC/Documents/GitHub/HW_7_TS_ML_Inputs-U10/Resources/yen.csv"
yen_futures = pd.read_csv(
    filepath, index_col="Date", infer_datetime_format=True, parse_dates=True
)
print(yen_futures.head())

#%%
