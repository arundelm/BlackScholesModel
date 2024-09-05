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
from math import sqrt, erf
import numpy as np
from scipy.stats import norm


def normal_distribution(x):
    return .5 * (1 + erf(x / sqrt(2)))

def d1(currentStockPrice, riskFreeRate, timeToExp, volatility, strikePrice):
    numerator = (np.log(currentStockPrice/strikePrice) + (riskFreeRate + (volatility*volatility/2)*timeToExp))
    denominator = volatility*sqrt(timeToExp)
    return numerator/denominator

def d2(currentStockPrice, riskFreeRate, timeToExp, volatility, strikePrice):
    return d1(currentStockPrice, riskFreeRate, timeToExp, volatility, strikePrice) - volatility*sqrt(timeToExp)

def callPrice(currentStockPrice, riskFreeRate, timeToExp, volatility, strikePrice):
    dOne = normal_distribution(d1(currentStockPrice, riskFreeRate,timeToExp, volatility, strikePrice))
    dTwo = normal_distribution(d2(currentStockPrice, riskFreeRate,timeToExp,volatility, strikePrice))
    return currentStockPrice*(dOne) - np.exp(-riskFreeRate * timeToExp)*strikePrice*dTwo


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
    pastDate = date.replace(year=date.month - 3)

    data = yf.download(threeMonthTicker, pastDate, date)
    return data['Close'].iloc[-1]

def getPrice(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period='1d')['Close'].iloc[-1]

def getDate(date):
    endDate = datetime.strptime(date, '%Y-%m-%d')
    currentDate = datetime.now()
    return (endDate - currentDate).days / 365.25
    

def getVolatility(ticker):
    current_day = datetime.now()
    data = yf.download(ticker, start=current_day.replace(year=current_day.year - 1), end=current_day)

    data['Daily_Return'] = data['Adj Close'].pct_change()

    daily_vol = data['Daily_Return'].std()
    return daily_vol * np.sqrt(252)


#GREEKS

#change in option's premium for every $1 change in the stock price
def delta(S, strike, t, r, vol):
    return norm.cdf(d1(S,r,t,vol,strike), 0, 1)

#change in option's delta for every $1 change in stock price
def gamma(S, strike, t, r, vol):
    return S*norm.cdf(d1(S,r,t,vol, strike), 0, 1) - strike*np.exp(-r*t)*norm.cdf(d2(S,r,t,vol, strike), 0, 1)

#change in the premium for every 1% change in volatility
def vega(S, strike, t, r, vol):
    return S*norm.pdf(d1(S,r,t,vol,strike), 0, 1)*np.sqrt(t)

#change in option's premium for every day that passes (time decay)
def theta(S, strike, t, r, vol):
    return -S*norm.pdf(d1(S,r,t,vol,strike), 0,1)*vol/(2*np.sqrt(t)) - r*strike*np.exp(-r*t)*norm.cdf(d2(S,r,t,vol,strike), 0, 1)

#changes in premium for every 1% change in risk-free rate
def rho(S, strike, t, r, vol):
    return strike*t*np.exp(-r*t)*norm.cdf(d2(S,r,t,vol,strike), 0, 1)

#Sample data
S = getPrice(ticker)
r = getRiskFreeRate()
t = getDate(date)
vol = getVolatility(ticker)

r = r / 100

main_df=calls.copy()
cols = ['lastTradeDate', 'lastPrice', 'volume', 'openInterest', 'contractSize', 'currency']
main_df.drop(columns=cols, inplace=True)
main_df['bsmValuation'] = main_df.apply(lambda row: callPrice(S, r, t, vol, row['strike']), axis=1)
print(main_df.head(10))

accuracy = 0
# Iterate over the dataframes row by row
for i in range(len(main_df)):
    diff = abs(main_df['bsmValuation'][i] - calls['lastPrice'][i])
    accuracy = accuracy + diff
accuracy = accuracy / len(main_df) * 100


greeks = main_df.copy()
greeks['delta'] = greeks.apply(lambda row: delta(S, row['strike'], t, r, vol), axis=1)
greeks['gamma'] = greeks.apply(lambda row: gamma(S, row['strike'], t, r, vol), axis=1)
greeks['vega'] = greeks.apply(lambda row: vega(S, row['strike'], t, r, vol), axis=1)
greeks['theta'] = greeks.apply(lambda row: theta(S, row['strike'], t, r, vol), axis=1)
greeks['rho'] = greeks.apply(lambda row: rho(S, row['strike'], t, r, vol), axis=1)

print(greeks.head())