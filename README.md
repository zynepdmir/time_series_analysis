# Time Series Analysis of Unemployment Rate in the US
This project analyzes the unemployment rate in the US from January 1980 to November 2024 using various statistical tests and forecasting models.

Key Steps:
Exploratory Analysis: ACF and PACF plots were examined.
Stationarity Tests: Multiple tests were applied to assess both regular and seasonal unit roots.
Train-Test Split: The last 12 observations were used as a test set for forecast accuracy evaluation.
Anomaly Detection: A sharp increase in 2020 was detected and replaced with a smoother pattern.
Transformations: A log transformation was applied, followed by differencing to remove stochastic trend and seasonality.

Modeling:
The best SARIMA model identified was SARIMA(3,1,3)(3,0,2)[12].
Diagnostic checks included the Portmanteau test, normality of residuals, Breusch-Godfrey test (for autocorrelation), and a heteroskedasticity test.
For volatility modeling, GARCH(2,1) with ARMA(3,2) errors performed best.

Forecasting:
Forecasts were generated using NNETAR, Prophet, TBATS, and ETS models.
Among them, the NNETAR model showed the best forecasting performance based on test accuracy.

The dataset: https://fred.stlouisfed.org/series/UNRATE



