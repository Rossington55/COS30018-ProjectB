import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import yfinance as yf
from collections import deque
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU

import numpy as np
import pandas as pd
import random
import datetime as dt
from task2 import shuffle_in_unison, get_index_at_date
from task4 import Layer

K_DAYS = 100
LOOKUP_STEP = 10
N_STEPS = 2
FEATURE_COLUMNS = ['Adj Close', 'Open', 'High', 'Low', 'Close', 'Volume']

def create_custom_model_multistep(sequence_length, feature_length, layers, k_days):
    model = Sequential()

    for i in range(len(layers)):
        layer_type = layers[i].type
        layer_size = layers[i].size

        if layer_type == 'LSTM':
            if i == 0:#First Layer
                model.add(LSTM(units=layer_size, return_sequences=True,batch_input_shape =(None,sequence_length,feature_length)))
            elif i == len(layers)-1:# Last Layer
                model.add(LSTM(units=layer_size, return_sequences=False))
            else:# All other layers (hidden layers)
                model.add(LSTM(units=layer_size, return_sequences=True))
        elif layer_type == 'GRU':
            if i == 0:#First Layer
                model.add(GRU(units=layer_size, return_sequences=True,batch_input_shape =(None,sequence_length,feature_length)))
            elif i == len(layers)-1:# Last Layer
                model.add(GRU(units=layer_size, return_sequences=False))
            else:# All other layers (hidden layers)
                model.add(GRU(units=layer_size, return_sequences=True))

        model.add(Dropout(0.3))

    model.add(Dense(k_days, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def load_data_multistep(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=None, split_by_ratio=None, feature_columns=['Adj Close', 'Open','High','Low','Close','Volume'], data_start='2020-01-01', data_end='2024-01-01',k_days=5):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        split_by_date (bool): whether we split the dataset into training/testing by date, setting it 
            to False will split datasets in a random way
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yfinance
        data_start (date): The start date to fetch data between
        data_end (date): The end date to fetch data between
    """
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = yf.download(ticker, data_start, data_end)
        df.to_csv("ticker_dowload.csv")
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")

    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()

    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    # add date as a column
    if "date" not in df.columns:
        df["date"] = df.index

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # For every row in the df
    targets = []
    for i in range(len(df.index)):
        #Add the Adj Close of that many days in the future (on top of the lookup step)
        targets.append(df["Adj Close"][i+lookup_step:i+lookup_step+k_days])
    
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    
    # drop NaNs
    df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    target_ptr = 0
    # Split the data into sequences
    for entry in df[feature_columns + ["date"]].values:
        sequences.append(entry)
        if len(sequences) == n_steps:
            if targets[target_ptr] is not None:
                sequence_data.append([np.array(sequences), targets[0]])
                target_ptr+=1

    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence
    
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    if split_by_ratio:
        # split the dataset into training & testing sets by a ratio of the total length (not randomly splitting)
        split_index = int((1 - split_by_ratio) * len(X))
        result["X_train"] = X[:split_index]
        result["y_train"] = y[:split_index]
        result["X_test"]  = X[split_index:]
        result["y_test"]  = y[split_index:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    elif split_by_date:
        split_index = get_index_at_date(X,split_by_date)
        if split_index > -1:
            result["X_train"] = X[:split_index]
            result["y_train"] = y[:split_index]
            result["X_test"]  = X[split_index:]
            result["y_test"]  = y[split_index:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])

    else:    
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=random.randrange(0,1), shuffle=shuffle)

    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    return result

if __name__ == "__main__":
    data = load_data_multistep(ticker="CBA.AX", data_start='2021-01-01',data_end='2024-01-20', lookup_step=LOOKUP_STEP,n_steps=N_STEPS, 
                     split_by_ratio=0.5, feature_columns=FEATURE_COLUMNS, k_days=K_DAYS)

    layer_info = [
        Layer('LSTM',20),
        Layer('LSTM',20),
    ]

    model = create_custom_model_multistep(sequence_length=2,feature_length=len(FEATURE_COLUMNS),layers=layer_info, k_days=K_DAYS)

    history = model.fit(data["X_train"], data["y_train"], 
                        batch_size=128, epochs=25, 
                        validation_data=(data["X_test"], data["y_test"]), 
                        verbose=1)

    predicted = model.predict(data["X_test"])
    last_y_predicted = data["column_scaler"]["Adj Close"].inverse_transform(predicted)[-1]
    last_y_actual = data["column_scaler"]["Adj Close"].inverse_transform(data["y_test"])[-1]


    plt.plot(last_y_actual,c='b')
    plt.plot(last_y_predicted,c='r')
    plt.ylabel("Price")
    plt.xlabel("Days")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()

