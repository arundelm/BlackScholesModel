#Using the BSM model
#5 inputs: strike price, current stock price, time to expiration, risk-free rate, volatility
# + type of option (call/put)
#Prices european options (US options can be exercised before exp date)
#predicts that price of heavily traded assets follows Geometric Brownian motion w/ const drift and volatility
"""
ASSUMPTIONS:
    no dividends are paid out,
    markets are random,
    no transaction costs,
    risk-free rate/volatility are known + constant,
    returns are normally distributed,
    option is European and can only be exercised at exp
"""

"""
FORMULA: 
    C = SN(d1) - Ke^(-rt) N(d2)

"""


#Black-Scholes
from math import sqrt
import numpy as np
from scipy.stats import norm


def d1(currentStockPrice, riskFreeRate, timeToExp, volatility, strikePrice):
    numerator = (np.log(currentStockPrice/strikePrice) + (riskFreeRate + (volatility*volatility/2)*timeToExp))
    denominator = volatility*sqrt(timeToExp)
    return numerator/denominator

def d2(currentStockPrice, riskFreeRate, timeToExp, volatility, strikePrice):
    return d1(currentStockPrice, riskFreeRate, timeToExp, volatility, strikePrice) - volatility*sqrt(timeToExp)

def callPrice(currentStockPrice, riskFreeRate, timeToExp, volatility, strikePrice):
    dOne = d1(currentStockPrice, riskFreeRate,timeToExp, volatility, strikePrice)
    dTwo = d2(currentStockPrice, riskFreeRate,timeToExp,volatility, strikePrice)
    return currentStockPrice*(norm(dOne)) - np.exp(-riskFreeRate * timeToExp)*strikePrice*norm(dTwo)

def putPrice(currentStockPrice, riskFreeRate, timeToExp, volatility, strikePrice):
    dOne = d1(currentStockPrice,riskFreeRate,timeToExp, volatility, strikePrice)
    dTwo = d2(currentStockPrice,riskFreeRate,timeToExp,volatility, strikePrice)
    return strikePrice*np.exp(-riskFreeRate*timeToExp)*norm(-dTwo) - currentStockPrice*norm(-dOne)


import yfinance as yf
import pandas as pd

#example ticker
ticker = 'NVDA'
stock = yf.Ticker(ticker)
date = stock.options[0]

def getOptionData(tickerSymbol, date):
    ticker = yf.Ticker(tickerSymbol)
    data = ticker.option_chain(date)
    return data.calls, data.puts

calls, puts = getOptionData(ticker, date)
calls = pd.DataFrame(calls)
calls.head()

from datetime import datetime
#Using the 3 month treasury bill rate for the risk-free rate
def getRiskFreeRate():
    threeMonthTicker = "^IRX"
    date = datetime.now()
    pastDate = date.replace(year=date.year - .25)

    data = yf.download(threeMonthTicker, pastDate, date)
    return date['Close'].iloc[-1]

def getPrice(ticker):
    stock = yf.Ticker(ticker)
    price = stock.info['regularMarketPrice']
    return price

def getDate(date):
    endDate = datetime.strptime(date, '%Y-%m-%d')
    currentDate = datetime.now()
    return (endDate - currentDate).days / 365.25
    

def getVolatility(ticker):
    today = datetime.now()