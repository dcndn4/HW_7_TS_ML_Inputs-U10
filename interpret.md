# Unit 10—time series hw

Use time-series tools to predict movements in the value of the Japanese yen versus USD.

## Starter code

This contains example calcs to use as a guide. Your actual output may differ.

[Time-Series Starter Notebook](Starter_Code/time_series_analysis.ipynb)

[Linear Regression Starter Notebook](Starter_Code/regression_analysis.ipynb)

[Yen Data CSV File](Starter_Code/yen.csv)

- - -

## Instructions

### Time-Series Forecasting.ipynb

Load USD-Yen exchange rate futures data. Apply time series analysis to see if there's any predictable behavior.

Do:

1. Decomposition using a Hodrick-Prescott Filter (Decompose the Settle price into trend and noise).
2. ARMA to forecast Returns.
3. ARIMA to forecast the Settle Price.
4. GARCH to forecasting Volatility.

Use results from the above to answer:

1. Should you buy the yen today?
2. Will yen risk increase/decrease?
3. Would you feel confident using these models to trade with?


### Linear Regression Forecasting.ipynb

Build a Scikit-Learn linear regression model to predict Yen futures ("settle") returns with *lagged* Yen futures returns and categorical calendar seasonal effects (e.g., day-of-week or week-of-year seasonal effects).

Do:

1. Prep the data (creating Returns and Lagged Returns, split into training and testing)
2. Fit a Linear Regression Model.
3. Predict using the testing data.
4. Out-of-sample performance.
5. In-sample performance.

Use results from the above to answer:

* Does this model perform better on in-sample data or out-of-sample data?

- - -

### Hints and Considerations

* Out-of-sample data = Testing data = data the model hasn't seen yet.
* In-sample data = training data = data the model was trained on.


### Submission

* Create Jupyter Notebooks for the analysis and host the notebooks on GitHub.

* Include a Markdown that summarizes your models and findings and include this report in your GitHub repo.

* Submit the link to your GitHub project to Bootcampspot.


---

© 2021 Trilogy Education Services
