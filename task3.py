import mplfinance as fplt
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import numpy as np

def format_dataframe(df, trading_days):
    # If window size is 1, no need for formatting
    if trading_days <= 1:
        return df

    # Group together the window of days
    # Use the first Open value
    # The high value is the maximum high value in the data range
    # The low value is the minimum low value in the data range
    # Use the final Close and Adjusted close values
    df_agg = df.groupby(pd.Grouper(freq=str(trading_days) + 'D')).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last",
                                                  "Adj Close": "last"})
    
    # Window size may not be a factor of the data size. Remove all leftover rows
    df_formatted = df_agg.dropna(how='all')
    # Assure correct labelling
    df_formatted.columns = ["Open", "High", "Low", "Close", "Adj Close"]
    return df_formatted

def get_data(ticker, start,end):
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = yf.download(ticker, start, end)
        df.to_csv("ticker_dowload.csv")
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")

    return df



def candle_plot(ticker,start_date='2024-07-01',end_date='2024-08-01',candle_size=1):
    #Fetch the data
    df = get_data(ticker, start_date, end_date)
    # Format the data into windows
    df_formatted = format_dataframe(df, candle_size) 
    
    fplt.plot(
            df_formatted,
            type='candle',
            style='charles',
            title=f'{ticker} Stock Price Candlestick Plot (Window: {candle_size} Days)',
            ylabel='Price ($)'
        )

def box_plot(ticker, start_date="2024-07-01", end_date="2024-08-01", window_size=1):
    # Fetch stock market data using yfinance
    df = get_data(ticker,start_date,end_date)
    
    # Create a DataFrame with rolling windows of data
    boxplot_data = [df['Adj Close'].values[i-window_size:i] for i in range(window_size, len(df) + 1)]

    # Filter out any windows that contain NaN values
    boxplot_data = [data for data in boxplot_data if pd.notna(data).all()]

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(boxplot_data, patch_artist=True, boxprops=dict(facecolor="lightblue"))

    plt.title(f'{ticker} Stock Price Boxplot (Window: {window_size} Days)')
    plt.xlabel(f'Time Windows ({window_size} days each)')
    plt.ylabel('Adjusted Close Price')
    plt.grid(True)

    # Show the plot
    plt.show()



candle_plot(ticker="CBA.AX", candle_size=3)
box_plot(ticker="CBA.AX", start_date="2024-07-01",end_date="2024-08-01",window_size=3)