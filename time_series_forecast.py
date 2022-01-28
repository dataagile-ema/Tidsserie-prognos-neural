from neuralprophet import NeuralProphet
import pandas as pd
from matplotlib import pyplot as plt

# Load the data
df = pd.read_csv('Data/tidsserie_SIR2.csv')

# Create the model
model = NeuralProphet()

# Fit the model
model.fit(df, freq="D")

df_future = model.make_future_dataframe(df, periods=30)

# Make a forecast
forecast = model.predict(df_future)

# Plot the forecast
fig_forecast = model.plot(forecast)
fig_components = model.plot_components(forecast)
fig_model = model.plot_parameters()
plt.show()





