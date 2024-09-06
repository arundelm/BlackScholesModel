# Black-Scholes Options Pricing Model

This project implements a Python-based Black-Scholes (BSM) options pricing model for European call options. 
It pulls real-time stock data, calculates the option prices and the Greeks, and evaluates the accuracy of the Black-Scholes model by comparing the calculated prices to the actual market prices, all of which are present in the **output.txt** file.
The project also generates a visual comparison of the Black-Scholes valuation against the real market prices in the **bsmplot.pdf** file.


## Features

- **Real-Time Stock Data**: Uses yfinance to retrieve real-time stock data and option chains.
- **Black-Scholes Pricing**: Implements the Black-Scholes model to calculate European call option prices.
- **Greeks Calculation**: Computes the Greeks (Delta, Gamma, Vega, Theta, Rho) to analyze option sensitivity.
- **Accuracy Testing**: Compares Black-Scholes valuations to real market prices, achieving 95% accuracy.
- **Plotting**: Generates a plot showing the Black-Scholes model valuations versus actual market prices.
