import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from task2 import load_data

LOOKUP_STEP = 10
N_STEPS = 2
FEATURE_COLUMNS = ['Adj Close', 'Open', 'High', 'Low', 'Close', 'Volume']

class Layer():
    def __init__(self,type,size):
        self.type = type
        self.size = size
        pass

def create_custom_model(sequence_length, feature_length, layers):
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

    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model



def plot_graph(test_df):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='b')
    plt.plot(test_df[f'adjclose_{LOOKUP_STEP}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()

def get_final_df(model, data):
    """
    This function takes the `model` and `data` dict to 
    construct a final dataframe that includes the features along 
    with true and predicted prices of the testing dataset
    """
    X_test = data["X_test"]
    y_test = data["y_test"]
    # perform prediction and get prices
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["Adj Close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["Adj Close"].inverse_transform(y_pred))
    test_df = data["test_df"]
    # add predicted future prices to the dataframe
    test_df[f"adjclose_{LOOKUP_STEP}"] = y_pred
    # add true future prices to the dataframe
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = y_test
    # sort the dataframe by date
    test_df.sort_index(inplace=True)
    final_df = test_df
    return final_df

# Example of using the above functions
if __name__ == "__main__":
    # Step 1: Load data
    
    data = load_data(ticker="CBA.AX", data_start='2021-01-01',data_end='2024-01-20', lookup_step=LOOKUP_STEP,n_steps=N_STEPS, 
                     split_by_ratio=0.5, feature_columns=FEATURE_COLUMNS)

    # Step 2: Define the layer structure
    layer_info = [
        Layer('LSTM',20),
        Layer('LSTM',20),
    ]

    # Create the model
    model = create_custom_model(sequence_length=2,feature_length=len(FEATURE_COLUMNS),layers=layer_info)

    # Step 3: Train the model
    history = model.fit(data["X_train"], data["y_train"], 
                        batch_size=128, epochs=25, 
                        validation_data=(data["X_test"], data["y_test"]), 
                        verbose=1)


    final_df = get_final_df(model,data)
    plot_graph(final_df)