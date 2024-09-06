#Black-Scholes

from math import sqrt, erf
import numpy as np
from scipy.stats import norm
import yfinance as yf
import pandas as pd
from datetime import datetime

##############

#Calculates the normal distribution function
def normal_distribution(x):
    return .5 * (1 + erf(x / sqrt(2)))

#Calculates d1, d2 which are probability factors used in the BSM
def d1(S, riskFreeRate, timeToExp, volatility, K):
    numerator = (np.log(S/K) + (riskFreeRate + (volatility*volatility/2)*timeToExp))
    denominator = volatility*sqrt(timeToExp)
    return numerator/denominator

def d2(S, riskFreeRate, timeToExp, volatility, K):
    return d1(S, riskFreeRate, timeToExp, volatility, K) - volatility*sqrt(timeToExp)

# Calculates the price of a European Call Option
def callPrice(S, riskFreeRate, timeToExp, volatility, K):
    dOne = normal_distribution(d1(S, riskFreeRate,timeToExp, volatility, K))
    dTwo = normal_distribution(d2(S, riskFreeRate,timeToExp,volatility, K))
    return S*(dOne) - np.exp(-riskFreeRate * timeToExp)*K*dTwo


##############

#Example Stock to test with
ticker = 'NVDA'
stock = yf.Ticker(ticker)
date = stock.options[1]

#Returns the soldified call option data for the entered ticker from yfinance
def getOptionData(tickerSymbol, date):
    ticker = yf.Ticker(tickerSymbol)
    data = ticker.option_chain(date)
    return data.calls

#Makes the call option data a dataframe with Pandas
calls = getOptionData(ticker, date)
calls = pd.DataFrame(calls)
calls.head(10)

###############

#Fetches the risk-free rate (3 month treasury bill rate)
def getRiskFreeRate():
    threeMonthTicker = "^IRX"
    date = datetime.now()
    pastDate = date.replace(year=date.month - 3)

    data = yf.download(threeMonthTicker, pastDate, date)
    return data['Close'].iloc[-1]

#Fetches the current price of the stock
def getPrice(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period='1d')['Close'].iloc[-1]

#Fetches the time to expiration in years
def getDate(date):
    endDate = datetime.strptime(date, '%Y-%m-%d')
    currentDate = datetime.now()
    time_to_exp = (endDate - currentDate).days / 365.25
    if time_to_exp == 0:
        time_to_exp = 1e-6 
    return time_to_exp
 
#Calculates the volatility based on historical data
def getVolatility(ticker):
    current_day = datetime.now()
    data = yf.download(ticker, start=current_day.replace(year=current_day.year - 1), end=current_day)
    data['Daily_Return'] = data['Adj Close'].pct_change()
    daily_vol = data['Daily_Return'].std()
    annual_vol = daily_vol * np.sqrt(252)
    return annual_vol

#Sample data for NVDA example
S = getPrice(ticker)
r = getRiskFreeRate() / 100
T = getDate(date)
vol = getVolatility(ticker)

###############

#Adjusts the call options dataframe to only include relevant data
main_df=calls.copy()
cols = ['lastTradeDate', 'bid', 'ask', 'percentChange', 'inTheMoney','volume', 'openInterest', 'contractSize', 'currency']
main_df.drop(columns=cols, inplace=True)
main_df['bsmValuation'] = main_df.apply(lambda row: callPrice(S, r,T, vol, row['strike']), axis=1)
main_df = main_df.iloc[:50]
pd.set_option('display.max_columns', None)

###############


#GREEKS

#Represents the change in option's premium for every $1 change in the stock price
def delta(S, K,T, r, vol):
    D1 = d1(S,r,T,vol,K)
    return norm.cdf(D1, 0, 1)

#Represents the change in option's delta for every $1 change in stock price
def gamma(S, K,T, r, vol):
    D1 = d1(S,r,T,vol,K)
    return norm.pdf(D1, 0, 1) / (S*vol*np.sqrt(T))
    
#Represents the change in the premium for every 1% change in volatility
def vega(S, K,T, r, vol):
    D1 = d1(S,r,T,vol,K)
    return S*norm.pdf(D1, 0, 1)*np.sqrt(T)*.01

#Represents the change in option's premium for every day that passes (time decay)
def theta(S, K,T, r, vol):
    D1 = d1(S,r,T,vol,K)
    D2 = d2(S,r,T,vol,K)
    return -S*norm.pdf(D1, 0,1)*vol/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(D2, 0, 1)

#Represents the change in premium for every 1% change in risk-free rate
def rho(S, K,T, r, vol):
    D2 = d2(S,r,T,vol,K)
    return K*T*np.exp(-r*T)*norm.cdf(D2, 0, 1)


greeks = main_df.copy()
greeks['delta'] = greeks.apply(lambda row: delta(S, row['strike'],T, r, vol), axis=1)
greeks['gamma'] = greeks.apply(lambda row: gamma(S, row['strike'],T, r, vol), axis=1)
greeks['vega'] = greeks.apply(lambda row: vega(S, row['strike'],T, r, vol), axis=1)
greeks['theta'] = greeks.apply(lambda row: theta(S, row['strike'],T, r, vol), axis=1)
greeks['rho'] = greeks.apply(lambda row: rho(S, row['strike'],T, r, vol), axis=1)

print(greeks)


#Calculate the percent error between the last price of each option and the BSM Valuation
error = 0
for i in range(len(main_df)):
    diff = abs(greeks['bsmValuation'][i] - greeks['lastPrice'][i])
    error = error + diff / calls['lastPrice'][i]
accuracy = error / len(main_df) * 100
print("Percent Error: " , accuracy , "%" )


#Plot the resulting points to show BSM valuation against the actual price
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(greeks['bsmValuation'], label='BSM Valuation', marker='o')
plt.plot(greeks['lastPrice'], label='Last Price', marker='x')
plt.xlabel('Option Index')
plt.ylabel('Price')
plt.title('BSM Valuation vs Last Price')
plt.legend()
plt.grid(True)
plt.show()