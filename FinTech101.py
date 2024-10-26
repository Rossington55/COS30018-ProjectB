
from task2 import load_data
from task4 import Layer, create_custom_model, get_final_df, plot_graph
from task5 import create_custom_model_multistep, load_data_multistep

# Data Info
TICKER = "CBA.AX"
DATA_START = "2021-01-01"
DATA_END = "2024-01-20"

# Data Formatting
LOOKUP_STEP = 10
N_STEPS = 2
SPLIT_BY_RATIO=0.5
FEATURE_COLUMNS = ['Adj Close', 'Open', 'High', 'Low', 'Close', 'Volume']
K_DAYS=100


# Training
BATCH_SIZE=128
EPOCHS=25


if __name__ == "__main__":
    # Step 1: Load data
    data = load_data(ticker=TICKER, data_start=DATA_START,data_end=DATA_END, lookup_step=LOOKUP_STEP,n_steps=N_STEPS, 
                    split_by_ratio=SPLIT_BY_RATIO, feature_columns=FEATURE_COLUMNS)

    # Step 2: Define the layer structure
    layer_info = [
        Layer('LSTM',20),
        Layer('LSTM',20),
    ]

    # Create the model
    model = create_custom_model(sequence_length=2,feature_length=len(FEATURE_COLUMNS),layers=layer_info)

    # Step 3: Train the model
    history = model.fit(data["X_train"], data["y_train"], 
                        batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_data=(data["X_test"], data["y_test"]), 
                        verbose=1)


    # Output/Plot the predicted value
    final_df = get_final_df(model,data)
    plot_graph(final_df)