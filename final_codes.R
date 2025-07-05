#### Time Series Plot and Interpretation  ####
library(ggplot2)
library(forecast)
unrate_data <- read.csv("C:/Users/zynep/Downloads/UNRATE.csv")
unrate <- ts(unrate_data[, 2], start = c(1948, 1), frequency = 12)
unrate_subset <- window(unrate, start = 1980)
autoplot(unrate_subset, main="Unemployment Rate in USA", y = "") + theme_bw()

#Time series plot shows that;
#- The series does not have an upward or downward trend in the long term, but there is a cyclical pattern over
#time, where unemployment rate rises and falls consistently.
#- The significant spike around 2020 implies an outlier which could be related to the COVID-19 Pandemic.
#- There is no seasonality observed in this plot.


#### Cross Validation ####
# The last 12 observations (test_set) are kept out of the analysis.
#We will use them later to measure the forecast
#accuracy of the models.

train_set <- window(unrate_subset, end = c(2023,11))
test_set <- window(unrate_subset, start = c(2023,12)) # 12 observations

#### Anomaly Detection and Cleaning ####
library(tibble)
library(gridExtra)
library(forecast)
train_set_cleaned <- tsclean(train_set)
p1 <- autoplot(train_set, main= "Before Cleaning Anomalies", y="unrate")+theme_bw()
p2 <- autoplot(train_set_cleaned, main="After Cleaning Anomalies", y="unrate")+theme_bw()
grid.arrange(p1,p2,nrow=2)
# The sharp increase in 2020 is now replaced with a smoother pattern.

#### Box-Cox Transformation Analysis ####
#The series may need transformation.
BoxCox.lambda(train_set_cleaned) 
# the lambda is close to zero, then apply log transformation
train_set_cleaned_transformed <- log(train_set_cleaned)
BoxCox.lambda(train_set_cleaned_transformed) 
#After the log transformation the Box-Cox lambda is approximately 1,
#which implies the transformation worked

#### Unit Root Tests ####
#Now that the series is cleaned and transformed, unit root tests can be applied.
p1 <- ggAcf(train_set_cleaned_transformed) +
  ggtitle("ACF of UNRATE") +
  theme_bw()

p2 <- ggPacf(train_set_cleaned_transformed) +
  ggtitle("PACF of UNRATE") +
  theme_bw()

grid.arrange(p1,p2,nrow=1)

#The slow linear decay in acf plot implies non-stationarity.
#KPSS test for the level
#H0: the series is stationary.
#H1: the series is not stationary.

library(tseries)
kpss.test(train_set_cleaned_transformed, null="Level")
# reject h0, series is not stationary

#We can apply few more unit root tests to see if the results are coherent.
#Hypotheses of ADF test for the unit root:
#  H0: the series have unit root
#  H1: the series do not have unit root


mean(train_set_cleaned_transformed)
# mean is not zero, then use "nc"
library(fUnitRoots)
adfTest(train_set_cleaned_transformed, type="nc")
# fail to reject h0, the series have unit root

#Then we can test which type of trend the series may have.
#KPSS test for the trend:
#  H0: the series has deterministic trend
#  H1: the series has stochastic trend

library(tseries)
kpss.test(train_set_cleaned_transformed, null = "Trend")
# reject h0, the series have stochastic trend

#Hypothesis of ADF test for the trend:
#  H0: the series have stochastic trend
#  H1: the series have deterministic trend
adfTest(train_set_cleaned_transformed, lags = 1, type="ct")
# fail to reject h0, the series have stochastic trend

library(pdR)
test_hegy <- HEGY.test(train_set_cleaned_transformed, 
                       itsd=c(1,1,0),regvar=0, 
                       selectlags=list(mode="aic", Pmax=12))

#Let’s check seasonal unit roots, too.
#HEGY test
#H0: the series has unit root
#H1: the series do not have unit root
test_hegy$stats
# hegy test says the series don't have seasonal unit root
# we have regular unit root
# (coherent with kpss and adf tests)

library(uroot)
ch.test(train_set_cleaned_transformed,type = "dummy",sid=c(1:12)) 
# fail to reject h0, Seasonal patterns are stable over time

library(forecast)
ocsb.test(train_set_cleaned_transformed)
# test stat is smaller than the critical value, then reject h0,
# the series don't have seasonal unit root

#### Removing the Trend ####
# since the series have stochastic trend, apply differencing
ndiffs(train_set_cleaned_transformed)

dif_train_set_cleaned_transformed <- diff(train_set_cleaned_transformed)
kpss.test(dif_train_set_cleaned_transformed, null = "Level")
# fail to reject h0, the diff'd series is stationary

adf.test(dif_train_set_cleaned_transformed)
# reject h0, the diff'd series don't have unit root

test_hegy <- HEGY.test(dif_train_set_cleaned_transformed, itsd=c(1,0,0),regvar=0, selectlags=list(mode="aic", Pmax=12))
test_hegy$stats
# the diff'd series have neither regular nor seasonal unit root

#### Checking the Plots ####
autoplot(dif_train_set_cleaned_transformed)
# the diff'd series is stationary around zero mean, variance seems stable

p1 <- ggAcf(dif_train_set_cleaned_transformed, lag.max=50) + 
  ggtitle("ACF of Diff'd Series") + theme_bw()

p2 <- ggPacf(dif_train_set_cleaned_transformed, lag.max=50) + 
  ggtitle("PACF of Diff'd Series") + theme_bw()

grid.arrange(p1,p2,nrow=1)

#### Model Identification ####
# significant spikes at seasonal lags 24 and 36 in ACF plot implies annual seasonality.
# also, PACF plot has significant spikes at seasonal lags 12 and 36.
# then suggested models have form SARIMA(p,1,q)(P,0,Q)12

#### Parameter Estimation ####
Arima(train_set_cleaned_transformed,
              order = c(3, 1, 3), seasonal=c(3,0,3), # all parameters are significant 
              method="CSS-ML") # other methods give na's'

Arima(train_set_cleaned_transformed,
              order = c(3, 1, 3), seasonal=c(3,0,2),
              method="ML") # all parameters are significant 

Arima(train_set_cleaned_transformed,
      order = c(3, 1, 3), seasonal=c(3,0,2),
      method="CSS") # sar3, sma2 are not significant

Arima(train_set_cleaned_transformed,
              order = c(3, 1, 3), seasonal=c(3,0,2),
              method="CSS-ML") # all parameters are significant

Arima(train_set_cleaned_transformed,
      order = c(3, 1, 3), seasonal=c(2,0,3),
      method="CSS-ML") # all parameters are significant

Arima(train_set_cleaned_transformed,
      order = c(3, 1, 3), seasonal=c(2,0,2),
      method="CSS-ML") # sar2, sma2 are not significant

Arima(train_set_cleaned_transformed,
      order = c(3, 1, 3), seasonal=c(2,0,1),
      method="CSS-ML") # sar2 is not significant

Arima(train_set_cleaned_transformed,
      order = c(3, 1, 3), seasonal=c(1,0,2),
      method="CSS") # other methods give na's
# sar1 and ar3 are not significant

Arima(train_set_cleaned_transformed,
      order = c(3, 1, 3), seasonal=c(1,0,1),
      method="CSS-ML") # ar3, ma3 are not significant

# then we have
fit1 <- Arima(train_set_cleaned_transformed,
             order = c(3, 1, 3), seasonal=c(3,0,3), 
             method="CSS-ML")

fit2 <- Arima(train_set_cleaned_transformed,
              order = c(3, 1, 3), seasonal=c(3,0,2),
              method="ML")

fit3 <- Arima(train_set_cleaned_transformed,
              order = c(3, 1, 3), seasonal=c(3,0,2),
              method="CSS-ML")

fit4 <- Arima(train_set_cleaned_transformed,
              order = c(3, 1, 3), seasonal=c(2,0,3),
              method="CSS-ML")

matrix(c(fit1$aic, fit2$aic, fit3$aic, fit4$aic, fit1$bic, fit2$bic, fit3$bic, fit4$bic,
         fit1$aicc, fit2$aicc, fit3$aicc, fit4$aicc), dimnames = list(c("fit1", "fit2", "fit3", "fit4"),
                                                                     c("AIC", "BIC","AICC")),  nrow=4)
# fit3, SARIMA(3,1,3)(3,0,2)12 is the best model

#### Diagnostic Checking ####
##### Portmanteau Lack of Fit Test #####
r <- resid(fit3)
Box.test(r, lag=50, type = c("Ljung-Box"))
# fail to reject h0, the residuals of fit3 are uncorrelated

d1 <- ggAcf(r, lag = 50) +
  ggtitle("ACF of Residuals") +
  theme_minimal() 

d2 <- ggPacf(r, lag = 50) +
  ggtitle("PACF of Residuals") +
  theme_minimal()

grid.arrange(d1, d2, nrow=1) 
# some spikes exceed the white noise bands,
# the residuals might be correlated 

checkresiduals(fit3, lag=50)
# on the other hand, formal test says fail to reject h0,
# the residuals are uncorrelated
# there's no outliers or pattern in the residuals vs time plot 

tsdiag(fit3)
# the residuals look uncorrelated here, too

##### Normality of the Residuals #####
ggplot(r, aes(sample = r)) +stat_qq()+geom_qq_line()+ggtitle("QQ Plot of the Residuals")+theme_minimal()
# QQ plot shows the residuals are normally distributed

ggplot(r,aes(x=r))+geom_histogram(bins=20)+geom_density()+ggtitle("Histogram of Residuals")+theme_minimal()
# histogram confirms normality

shapiro.test(r)
# fail to reject h0, residuals are normal


##### Breusch-Godfrey test #####
library(TSA)
library(lmtest)
m <- lm(r ~ 1+zlag(r))
bgtest(m,order=15)
# fail to reject h0, the residuals are uncorrelated

##### Heteroskedasticity Test #####
rr <- r^2
g1 <- ggAcf(rr, lag.max = 50) + ggtitle("ACF of Squared Residuals") + theme_minimal()
g2 <- ggPacf(rr, lag.max = 50) + ggtitle("PACF of Squared Residuals") + theme_minimal()   
grid.arrange(g1,g2,ncol=2) 
# there might be heteroskedasticity problem

library(MTS)
archTest(r)
# ARCH Engle's test says reject h0, there exists ARCH effects
# Rank based test is borderline significant
# then consider GARCH models

##### GARCH Models #####
# what we want:
# * significant parameters
# * low info criteria
# * fail to reject Ljung Box test
# * fail to reject ARCH LM test
# * fail to reject Nyblom stability test
# * fail to reject sign bias test
# * fail to reject Pearson Goodness of fit test

# checking GARCH(1,1)
library(rugarch)
spec <- ugarchspec(variance.model = list(model="sGARCH")) 
def.fit <- ugarchfit(spec, train_set_cleaned)
print(def.fit)
# significance of some parameters and pearson test problem

# checking GARCH(2,1)
spec2 <- ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(2,1))) 
def.fit2 <- ugarchfit(spec2, train_set_cleaned)
print(def.fit2)
# significance of some parameters and pearson test problem

spec2 <- ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,2))) 
def.fit2 <- ugarchfit(spec2, train_set_cleaned)
print(def.fit2)
# significance of some parameters, ljung-box and pearson test problem

spec2 <- ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(2,2))) 
def.fit2 <- ugarchfit(spec2, train_set_cleaned)
print(def.fit2)
# significance of some parameters and pearson test problem

spec3 <- ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)), 
                    mean.model=list(armaOrder=c(3,3),
                                    include.mean=TRUE))
def.fit3 <- ugarchfit(spec3, train_set_cleaned)
print(def.fit3)
# significance of some parameters problem

spec3 <- ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(2,1)), 
                    mean.model=list(armaOrder=c(3,3),
                                    include.mean=TRUE))
def.fit3 <- ugarchfit(spec3, train_set_cleaned)
print(def.fit3)
# significance of some parameters problem

spec3 <- ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,2)), 
                    mean.model=list(armaOrder=c(3,3),
                                    include.mean=TRUE))
def.fit3 <- ugarchfit(spec3, train_set_cleaned)
print(def.fit3)
# significance of some parameters problem

spec3 <- ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(2,2)), 
                    mean.model=list(armaOrder=c(3,3),
                                    include.mean=TRUE))
def.fit3 <- ugarchfit(spec3, train_set_cleaned)
print(def.fit3)
# significance of some parameters problem

spec3 <- ugarchspec(variance.model=list(model="apARCH")) 
def.fit3 <- ugarchfit(spec3, train_set_cleaned)
print(def.fit3)
# significance of some parameters and pearson test problem

spec3 <- ugarchspec(variance.model=list(model="apARCH", garchOrder=c(1,2))) 
def.fit3 <- ugarchfit(spec3, train_set_cleaned)
print(def.fit3)
# significance of some parameters and pearson test problem

spec3 <- ugarchspec(variance.model=list(model="apARCH", garchOrder=c(2,1))) 
def.fit3 <- ugarchfit(spec3, train_set_cleaned)
print(def.fit3)
# significance of so many parameters, nyblom joint test and pearson test problem

spec3 <- ugarchspec(variance.model=list(model="apARCH", garchOrder=c(2,2))) 
def.fit3 <- ugarchfit(spec3, train_set_cleaned)
print(def.fit3)
# significance of so many parameters, nyblom joint test and pearson test problem

spec3 <- ugarchspec(variance.model=list(model="apARCH", garchOrder=c(1,1)), 
                    mean.model=list(armaOrder=c(3,3),
                                    include.mean=TRUE))
def.fit3 <- ugarchfit(spec3, train_set_cleaned)
print(def.fit3)
# significance of some parameters and nyblom joint test problem

spec3 <- ugarchspec(variance.model=list(model="apARCH", garchOrder=c(1,2)), 
                    mean.model=list(armaOrder=c(3,3),
                                    include.mean=TRUE))
def.fit3 <- ugarchfit(spec3, train_set_cleaned)
print(def.fit3)
# significance of some parameters and nyblom joint test problem

spec3 <- ugarchspec(variance.model=list(model="apARCH", garchOrder=c(2,1)),
                    mean.model=list(armaOrder=c(3,3),
                                    include.mean=TRUE))
def.fit3 <- ugarchfit(spec3, train_set_cleaned)
print(def.fit3)
# gives NaN

spec3 <- ugarchspec(variance.model=list(model="apARCH", garchOrder=c(2,2)), 
                    mean.model=list(armaOrder=c(3,3),
                                    include.mean=TRUE))
def.fit3 <- ugarchfit(spec3, train_set_cleaned)
print(def.fit3)
# significance of some parameters and nyblom joint test problem

# then we can use GARCH(2,1) model with ARMA(3,3) 
# but let's check GARCH(2,1) with ARMA(3,2), too

spec3 <- ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(2,1)), 
                    mean.model=list(armaOrder=c(3,2),
                                    include.mean=TRUE))
def.fit3 <- ugarchfit(spec3, train_set_cleaned)
print(def.fit3)
# more parameters are significant here and all of the assumptions are satisfied.
# then we continue with GARCH(2,1) with ARMA(3,2)

#### Forecasting ####
# minimum mse forecast 
spec <- ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(2,1)), 
                    mean.model=list(armaOrder=c(3,2),
                                    include.mean=TRUE))
def.fit <- ugarchfit(spec, train_set_cleaned)
boot <- ugarchboot(def.fit,
                   method=c("Partial","Full")[1],
                   n.ahead = 12,
                   n.bootpred=1000,
                   n.bootfit=1000)
f <- boot@forc@forecast$seriesFor
f_vec <- as.vector(f)

##### ets forecast #####
ets_model <- ets(train_set_cleaned)
ets_model
checkresiduals(ets_model)
# it say multiplicative error, additive trend but no seasonality.
# so the ets model couldn't capture the seasonality in the data

# then use Holt Winter's method
hw_model_add <- hw(train_set_cleaned, seasonal="additive", h=12)
hw_model_mult <- hw(train_set_cleaned, seasonal="multiplicative", h=12)

summary(hw_model_add)
summary(hw_model_mult)

checkresiduals(hw_model_add)
checkresiduals(hw_model_mult)
# both ACF plots are far from white noise behaviour

##### nnetar forecast #####
nn_model <- nnetar(train_set_cleaned)
nn_model
checkresiduals(nn_model)
# time series plot shows that nnetar model captures the trend and seasonality of the data
# ACF plot shows all strikes are within white noise bands, except 1 spike
# histogram shows that the residuals are normally distributed
# Ljung-Box test is failed to reject, then the residuals are not correlated

##### TBATS forecast #####
tbats_model <- tbats(train_set_cleaned)
checkresiduals(tbats_model)
# no trend or seasonality in the residuals plot 
# ACF plot shows some spikes exceeding white  noise bands
# histogram shows normal behaviour
# Ljung-Box test is rejected, the residuals are not correlated

##### prophet forecast #####
library(prophet)
ds<-c(seq(as.Date("1980/01/01"), as.Date("2023/11/01"),by="month"))
df<-data.frame(ds, y=as.numeric(train_set_cleaned))
fit_prophet <- prophet(df)
future <- make_future_dataframe(fit_prophet,periods = 12)
prophet_f <- predict(fit_prophet, future)

resid <- df$y - prophet_f$yhat[1:length(df$y)] # not sure
checkresiduals(resid)
# none of the assumptions are satisfied
# prophet couldn't capture the information in the data

#### Accuracy #####
spec <- ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(2,1)), 
                    mean.model=list(armaOrder=c(3,2),
                                    include.mean=TRUE))
def.fit <- ugarchfit(spec, train_set_cleaned)
f <- ugarchforecast(def.fit, n.ahead=12)
s <- as.vector(f@forecast$seriesFor)

nn_f <- forecast(nn_model, h=12, PI=TRUE)

tbats_f <- forecast(tbats_model, h=12)

accuracy(s, test_set)
accuracy(hw_model_add, test_set)
accuracy(hw_model_mult, test_set)
accuracy(nn_f, test_set)
accuracy(tbats_f, test_set)
accuracy(tail(prophet_f$yhat, 12), test_set)

# nnetar model might have overfitting because it explains the train set a lot better than test set
# prophet model's error values are too high, so it underperforms
# the best model is Holt Winter's with multiplicative seasonality

#### Forecast Plots ####
library(ggplot2)
ts_garch <- ts(s, frequency=12, start=c(2023,12)) 
autoplot(test_set, series="Series") +
  geom_vline(xintercept = as.Date("2023-12-01"), color = "red")

hw_mult_f <- forecast(hw_model_mult, h=12)
hw_add_f <- forecast(hw_model_add, h=12)

autoplot(train_set, series="Series", main="NNETAR Forecast") +
  autolayer(fitted(nn_model), series="Fitted values") + 
  autolayer(nn_f$mean, series="Point forecast") +
  geom_vline(xintercept=2023+11/12, color="red") +
  theme_bw()

autoplot(train_set, series="Series", main="TBATS Forecast") + 
  autolayer(fitted(tbats_model), series="Fitted values") +
  autolayer(tbats_f$mean, series="Point forecast") +
  geom_vline(xintercept=2023+11/12, color="red") +
  theme_bw()

autoplot(train_set, series="Series", main="Holt Winter's Multiplicative Forecast") + 
  autolayer(fitted(hw_model_mult), series="Fitted values") +
  autolayer(hw_model_mult$mean, series="Point forecast") +
  geom_vline(xintercept=2023+11/12, color="red") +
  theme_bw()
