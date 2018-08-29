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

ARIMA is acronym for AR, I, and MA.

```
AR: Autoregression. A model that uses the dependent relationship between an observation and some number of lagged observations.
I: Integrated. The use of differencing of raw observations (e.g. subtracting an observation from an observation at the previous time step) in order to make the time series stationary.
MA: Moving Average. A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
```
### Time Decomposition
Time decomposition charts are useful to get an initial feel of the time series data. Here, the additive Time Decomposition chart shows some trend, seasonality and noise detected in the time series data. 

![timedecomp1](/image/arima/timedecomp1.png)

Next, I did a first order differencing and tested for stationarity in the time series data. p < 0.05, hence we can conclude the differenced time series is now statationary.

![stationarity](/image/arima/stationarity.png)

### ACF and PACF
The ACF plot is merely a bar chart of the coefficients of correlation between a time series and lags of itself. The PACF plot is a plot of the partial correlation coefficients between the series and lags of itself. By looking at the autocorrelation function (ACF) and partial autocorrelation (PACF) plots of the differenced series, i can tentatively identify the numbers of AR and/or MA terms that are needed. 

![acfpacf](/image/arima/acfpacf.png)

### Fit and plot the ARIMA model
#### In-sample prediction

```
ar1ma1 = ARMA(series_diff, (1, 1)).fit()
ar1ma1.summary()
```

![arima_summary](/image/arima/arima_summary.png)

An ARIMA model can be created as follow:

Define the model by calling ARMA() and passing in the p, and q parameters.
The model is prepared on the training data by calling the fit() function.
The training data is the differenced series. In this case, this is a first-differenced series. 
Predictions can be made by calling the predict() function and specifying the index of the time or times to be predicted.
`predictions = ar1ma1.predict()`

The below plot compares the predicted series with first-differenced series that was fed into the model.
![series_diff](/image/arima/series_diff.png)

The below plot compares the predicted full signal with the original full signal. 
![fullpred](/image/arima/fullpred.png)

The predicted signal looked promising, following the supposed trend, however, it could have been better. This is because if ether holder were to sell off earlier near the predicted peak, they would have missed out on the huge bull run which came after. 

#### Out-of-sample prediction

Out-of-sample prediction starts at row 100 and **dynamic** is set to **True**. The dynamic keyword affects in-sample prediction. If dynamic is False, then the in-sample lagged values are used for prediction. If dynamic is True, then in-sample forecasts are used in place of lagged dependent variables. The first forecasted value is start.

`predictions_oos = ar1ma1.predict(start=100, dynamic=True)`

A plot of the out-of-sample prediction is done below,

![oos_full](/image/arima/oos_full.png)

We can observe that the model fit is relatively good, yet the forecast is very weak. In fact, the forecast quickly reverts to the mean. It is important to note that a good in-sample fit does not necessarily equate to a good forecast, and vice-versa.

The reason that forecasting fails here is that there was no consideration during model fitting to identify the key drivers of the time series. 

Hence, rather than manually grid-search through the various permutation of parameters, I have also tried letting the machine do the hard work of finding the optimal parameters for us as shown next. 



## SARIMAX model

For this model, I demonstrated with the weekly closing price and normalized it through a Box-Cox transformation.

![weekclose](/image/sarimax/weekclose.png)

```
def box_cox_rolling_coeffvar(box_cox_param, endog, freq):
    """helper to find Box-Cox transformation with constant standard deviation
    
    returns RLM results instance
    """
    roll_air = special.boxcox(endog, box_cox_param).rolling(window=freq)
    y = roll_air.std() 
    m = roll_air.mean()
    x = sm.add_constant(m)
    res_rlm = sm.RLM(y, x, missing='drop').fit()
    return res_rlm

endog = df_weeklyclose['close']
freq = 4
tt = [(lam, box_cox_rolling_coeffvar(lam, endog, freq).pvalues[1]) for lam in np.linspace(-1, 1, 21)]

tt = np.asarray(tt)
print(tt[tt[:,1].argmax()])

```
```
from scipy import special

ax = special.boxcox(df_weeklyclose['close'], 0).rolling(window=4).std().plot(figsize=(12,6))
```

![boxcox](/image/sarimax/boxcox.png)


I have also tried using Auto-Arima from the **pyramid.arima** library to help me in finding the optimal model in this approach. 
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

The output of our code suggests that using the model SARIMAX(2, 1, 0)x(0, 1, 1, 12) which yields the lowest AIC value of -97.012. We should therefore consider this to be optimal option out of all the models we have generated.

### What is the Akaike Information Criterion (AIC)?

The Akaike information criterion (AIC) is an estimator of the relative quality of statistical models for a given set of data. Given a collection of models for the data, AIC estimates the quality of each model, relative to each of the other models.

A summary report of the stepwise_model:

`stepwise_model.summary()`

![sarimaxsummary1](/image/sarimax/sarimaxsummary1.png)

### Interpreting the summary report

The **coef** column shows the weight (i.e. importance) of each feature and how each one impacts the time series. 
The **P>|z|** column informs us of the significance of each feature weight. Here, each weight has a p-value lower or close to 0.05, so it is reasonable to retain all of them in our model. 
The **Jarque-Bera** test is a goodness-of-fit test of whether the data has the skewness and kurtosis of a normal distribution. The normal distribution has a skew of **0.22** and a kurtosis of **3.67**.

### Plot model diagnostics
When fitting seasonal ARIMA models (and any other models for that matter), it is important to run model diagnostics to ensure that none of the assumptions made by the model have been violated. The plot_diagnostics object allows us to quickly generate model diagnostics and investigate for any unusual behavior.

![plotdiag](/image/sarimax/plotdiag.png)

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

![insample1](/image/sarimax/insample1.png)

Prediction quality: 1.73 MSE (1.31 RMSE) 
Very good!


### Sarimax Out-Sample Prediction

Forecasting for the next 3 months,

```
pred = res_s.get_prediction(start=pd.to_datetime('2018-08-19'), end=pd.to_datetime('2018-12-19'))
pred_ci = pred.conf_int()

```

After doing a reverse transformation, 

![outsample1](/image/sarimax/outsample1.png)

### Notes: 
1. The **get_prediction** and **conf_int** methods calculate predictions for future points in time for the previously fitted model and the confidence intervals associated with a prediction, respectively. The **dynamic=False** argument causes the method to produce a one-step ahead prediction of the time series.
2. If you are forecasting for an observation that was part of the data sample - it is **in-sample** forecast. If you are forecasting for an observation that was not part of the data sample - it is **out-of-sample** forecast.
3. As we forecast further out into the future, it is natural for the model to become less confident in its values. This is reflected by the confidence intervals generated by our model, which grow larger as we move further out into the future.


### Multilayer Perceptron Network
A perceptron is a single neuron model that was a precursor to larger neural networks.
The building block for neural networks are artificial neurons. These are simple computational units that have weighted input signals and produce an output signal using an activation function.

#### What is an activation function?
An activation function is a simple mapping of summed weighted input to the output of the neuron. It is called an activation function because it governs the threshold at which the neuron is activated and strength of the output signal. Recently the rectifier activation function (RELU) has been shown to provide better results.



![neuron](/image/mlp/neuron.png)

#### Training and Testing datasets
I feed the network with data of daily closing prices. 

![dayclose](/image/mlp/dayclose.png)

Then the data is split into 80% training set and 20% testing set.

`split_point = int(len(data)*0.8)`
`train = data[:split_point]`
`test = data[split_point:]`

#### Data preparation stage
The data preparation stage is an important first step. Data must be in numerical values.
Neural networks require the input to be scaled in a consistent way. You can rescale it to the range between 0 and 1 called normalization. Another popular technique is to standardize it so that the distribution of each column has the mean of zero and the standard deviation of 1.

```
lags = 1
X_train, y_train = prepare_data(train, lags)
X_test, y_test = prepare_data(test, lags)
y_true = y_test     # due to naming convention
```

#### Training the neural network
This simple network will have one input (size of the lags variable), one hidden layer with 3 neurons and an output layer. The model is fitted using the MSE criterion and rectified linear units (RELU) as activation function. We instantiate the Sequential class, successively add the layer containing the neurons and specify the specific type of activation function within the Dense class (which correspond to the neuron. Then we specify a loss function and an optimizer to be used to adjust the networks neuron weights iteratively. Lastly, the model is fitted to the training data set. The model is trained over 20 epochs.

```
mdl = Sequential()
mdl.add(Dense(3, input_dim=lags, activation='relu'))
mdl.add(Dense(1))
mdl.compile(loss='mean_squared_error', optimizer='adam')
mdl.fit(X_train, y_train, epochs=20, batch_size=2, verbose=2)
```

#### Model Accuracy
We then check for the training and test accuracy of the sample using the evaluate method. Then we call the prediction method on our model and prepare the data for plotting. 

Train Score: 9.31 MSE (3.05 RMSE)
Test Score: 6.34 MSE (2.52 RMSE)

![quadpred1](/image/mlp/quadpred1.png)

#### Prediction
Once a neural network has been trained it can be used to make predictions. Since the neural network has only been fed by the last observation, it did not have much choice but to learn to apply observation t for the prediction of t+1. 

![pred1](/image/mlp/pred1.png)

#### MLP with window
1. We improve prediction by increasing the back-looking windows(lags).
2. Increase the number of hidden layers as well as the number of neurons for each layer.

```
# create and fit Multilayer Perceptron model
mdl = Sequential()
mdl.add(Dense(4, input_dim=lags, activation='relu'))
mdl.add(Dense(8, activation='relu'))
mdl.add(Dense(1))
mdl.compile(loss='mean_squared_error', optimizer='adam')
mdl.fit(X_train, y_train, epochs=20, batch_size=2, verbose=2)
 
# Estimate model performance
train_score = mdl.evaluate(X_train, y_train, verbose=0)
print('Train Score: {:.2f} MSE ({:.2f} RMSE)'.format(train_score, math.sqrt(train_score)))
test_score = mdl.evaluate(X_test, y_test, verbose=0)
print('Test Score: {:.2f} MSE ({:.2f} RMSE)'.format(test_score, math.sqrt(test_score)))
```

Train Score: 17.76 MSE (4.21 RMSE)
Test Score: 17.53 MSE (4.19 RMSE)

![quadpred2](/image/mlp/quadpred2.png)

The accuracy for our test set has improved compared to the first neural network we implemented. This is on account of the network utilizing a longer period of back-looking data to conduct its prediction.

![pred2](/image/mlp/pred2.png)

### Long Short Term Memory Network (LSTM)

A Long short-term memory (LSTM) is a type of Recurrent Neural Network specially designed to prevent the neural network output for a given input from either decaying or exploding as it cycles through the feedback loops. Long short-term memory networks provide better performance compared to other RNN architectures by alleviating what is called the vanishing gradient problem.

![lstm](/image/lstm/lstm.png)

The 5-mins interval data was normalized using MinMaxScaler, 
```
mms = MinMaxScaler(feature_range=(-1,1))
eth_norm = pd.DataFrame(mms.fit_transform(df_eth_close), columns=df_eth_close.columns)

```

![minmax](/image/lstm/minmax.png)

The dataset is split into 80% training set and 20% test set. Then we create a LSTM model.

```
model = Sequential()
# Remember: "batch_input_shape" and specify the batch size
model.add(LSTM(32, batch_input_shape=(1, 1, Xtrain.shape[2]), stateful=True))
model.add(Dense(Xtrain.shape[2]))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
```

After creating the model, I trained the model on the training data set over 20 epochs.
```
for i in range(20):
    if (i % 5) == 0:
        print 'ITER:', i
        model.fit(Xtrain, Ytrain, nb_epoch=1, batch_size=1, verbose=1, shuffle=False)
    else:
        model.fit(Xtrain, Ytrain, nb_epoch=1, batch_size=1, verbose=0, shuffle=False)
    model.reset_states()
``` 

A plot of the prediction against the actual, with **R2 score = 0.63** looks pretty promising.

![lstmpred](/image/lstm/lstmpred.png)


## Next steps
#### Hyperparamters tuning
Hyperparameter tuning can be explored, we can take note of the following points if we were to manually optimize hyperparameters for RNNs.

1. Watch out for overfitting. Overfitting means you get great performance on training data, but the network’s model is useless for out-of-sample prediction. 
2. Regularization helps: regularization methods include l1, l2, and dropout among others. 
3. More data is almost always better, because it helps fight overfitting. 
4. Train over multiple epochs (complete passes through the dataset). Evaluate test set performance at each epoch to know when to stop (early stopping). 
5. The learning rate is the single most important hyperparameter. 

## In Conclusion
To predict the future price of cryptocurrencies is tougher than it looks. I've tried Arima, Sarimax, MLPN and LSTM. In some cases the results look promising. But there are also alot more details that can still be improved.
    
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
