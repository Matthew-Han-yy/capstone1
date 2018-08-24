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
