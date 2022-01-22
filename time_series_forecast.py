from neuralprophet import NeuralProphet
import pandas as pd

# Load the data
df = pd.read_csv('Data/tidsserie_SIR2.csv', index_col='Month')

# Create the model
model = NeuralProphet(
    hidden_layers=[10, 10],
    loss='mse',
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=True
)

# Fit the model
model.fit(df)

# Make a forecast
forecast = model.predict(df, steps=15)

# Plot the forecast
model.plot(forecast)

