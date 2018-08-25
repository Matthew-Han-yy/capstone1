## Ethereum Price Trend Prediction Analysis 
Ethereum is a decentralized platform that runs smart contracts: applications that run exactly as programmed without any possibility of downtime, censorship, fraud or third-party interference. These apps run on a custom built blockchain, an enormously powerful shared global infrastructure that can move value around and represent the ownership of property.

The goal of this project is not giving readers about the screws and bolts of the technology behind Ethereum but more about its price trend prediction, thus being more investor-friendly. Here we go!

### Where I got my ether price data
I highly recommend https://poloniex.com/. It provides almost real-time price data on various digital currency including ether. My project pulls price data of ether with 5 minutes interval. You may also wish to customize the interval to your liking. Below is a chart of Ether closing price from its inception until now. 

![ethcloseprice](/image/ethcloseprice.png)

Below is a rolling mean chart. 

![rollingmeanchart](/image/rollingmeanchart.png)


## Models I used
I tried out the ARIMA, SARIMAX models for statsmodels and for the deep learning part, I tried out Multilayer Perceptron network (MLP) and Long Short Term Memory models (LSTM).

## ARIMA model

Time decomposition charts are useful to get an initial feel of the time series data. Here, the additive Time Decomposition chart shows some trend, seasonality and noise detected in the time series data. 

![timedecomp1](/image/arima/timedecomp1.png)

Next, I did a first order differencing and tested for stationarity in the time series data. p < 0.05, hence we can conclude the differenced time series is now statationary.

![stationarity](/image/arima/stationarity.png)

### ACF and PACF
The ACF plot is merely a bar chart of the coefficients of correlation between a time series and lags of itself. The PACF plot is a plot of the partial correlation coefficients between the series and lags of itself. By looking at the autocorrelation function (ACF) and partial autocorrelation (PACF) plots of the differenced series, i can tentatively identify the numbers of AR and/or MA terms that are needed. 

![acfpacf](/image/arima/acfpacf.png)

### Fit and plot the ARIMA model


## SARIMAX model

For this model, I demonstrated with the weekly closing price and normalized it through a Box-Cox transformation.

![weekclose](/image/sarimax/weekclose.png)

![boxcox](/image/sarimax/boxcox.png)


I also tried using Auto-Arima from the **pyramid.arima** library to help me in finding the optimal model in this approach. 
```
# Auto Arima
stepwise_model = auto_arima(weeklyclose_t, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())
```

The output of auto-arima suggests that using the model SARIMAX(1, 1, 1)x(1, 1, 1, 12) which yields the lowest AIC value of 1600.029. Therefore, this is to be optimal option out of all the models.

A summary report of the stepwise_model:

`stepwise_model.summary()`

![sarimaxsummary](/image/sarimax/sarimaxsummary.png)

### Interpreting the summary report

The **coef** column shows the weight (i.e. importance) of each feature and how each one impacts the time series. 
The **P>|z|** column informs us of the significance of each feature weight. Here, each weight has a p-value lower or close to 0.05, so it is reasonable to retain all of them in our model. 
The **Jarque-Bera** test is a goodness-of-fit test of whether the data has the skewness and kurtosis of a normal distribution. The normal distribution has a skew of **1.35** and a kurtosis of **4.96**.

### Plot model diagnostics
When fitting seasonal ARIMA models (and any other models for that matter), it is important to run model diagnostics to ensure that none of the assumptions made by the model have been violated. The plot_diagnostics object allows us to quickly generate model diagnostics and investigate for any unusual behavior.

![plotdiagnostics](/image/sarimax/plotdiagnostics.png)

We need to ensure that the residuals of our model are randomly distributed with zero-mean and not serially correlate, i. e. we’d like the remaining information to be white noise. If the fitted seasonal ARIMA model does not satisfy these properties, it is a good indication that it can be further improved.
The **residual** plot of the fitted model in the upper left corner appears do be white noise as it does not display obvious seasonality or trend behaviour. The **histogram** plot in the upper right corner pair with the kernel density estimation (red line) indicates that the time series is almost normally distributed. This is compared to the density of the standard normal distribution (green line). The **correlogram** (autocorrelation plot) confirms this resuts, since the time series residuals show low correlations with lagged residuals.
Although the fit so far appears to be fine, a better fit could be achieved with a more complex model.

### Sarimax In-Sample Prediction
```
pred = res_s.get_prediction(start=pd.to_datetime('2018-01-07'), 
                          end=pd.to_datetime('2018-08-19'),
                          dynamic=True, full_results=True)

pred_ci = pred.conf_int()

```

![insample](/image/sarimax/insample.png)

Prediction quality: 283.56 RMSE


### Sarimax Out-Sample Prediction
```
pred = res_s.get_prediction(start=pd.to_datetime('2018-08-19'), end=pd.to_datetime('2018-12-19'))
pred_ci = pred.conf_int()

```

![outsample](/image/sarimax/outsample.png)

### Notes: 
1. The **get_prediction** and **conf_int** methods calculate predictions for future points in time for the previously fitted model and the confidence intervals associated with a prediction, respectively. The **dynamic=False** argument causes the method to produce a one-step ahead prediction of the time series.
2. If you are forecasting for an observation that was part of the data sample - it is **in-sample** forecast. If you are forecasting for an observation that was not part of the data sample - it is **out-of-sample** forecast.
3. As we forecast further out into the future, it is natural for the model to become less confident in its values. This is reflected by the confidence intervals generated by our model, which grow larger as we move further out into the future.











### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Matthew-Han-yy/capstone1/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
