
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from task5 import N_STEPS, create_custom_model_multistep, load_data_multistep, FEATURE_COLUMNS, LOOKUP_STEP, K_DAYS, Layer

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

lstm_predicted = model.predict(data["X_test"])
last_lstm_prediction = data["column_scaler"]["Adj Close"].inverse_transform(lstm_predicted)[-1]
last_y_actual = data["column_scaler"]["Adj Close"].inverse_transform(data["y_test"])[-1]

def fit_arima(train_data,n_steps,integrated,moving_average ): 
    model = ARIMA(train_data, order=(n_steps,integrated,moving_average))
    arima_model = model.fit()
    return arima_model

# Use only 'Adj Close' column for ARIMA
adj_close_series = data['df']['Adj Close']

arima_model = fit_arima(adj_close_series,n_steps=2,integrated=0,moving_average=0)
arima_predictions = arima_model.forecast(steps=1)

        
print("Actual Final Day Close: ",last_y_actual[-1])
ensemble_predictions = (last_lstm_prediction[-1] + arima_predictions) / 2
print("Predicted Final Day Close: ",ensemble_predictions)
print(arima_predictions)
print(last_lstm_prediction[-1])




