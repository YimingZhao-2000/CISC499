import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


# Load dataset
data = pd.read_csv("../dataset/EA_dataset.csv")
# print(data.head()) # See if read successfully
# print(data.shape) # (row, column)


# Split dataset into training and testing
data["Date"] = pd.to_datetime(data["Date"])
train_data = data[data["Date"] < '2018-01-01']
test_data = data[data["Date"] >= '2018-01-01']


# Prepare the data for Prophet
train_data = train_data.rename(columns={"Date": "ds", "Close_Price": "y"})


# Train the model
model = Prophet()
model.fit(train_data)


# Make predictions
future = model.make_future_dataframe(periods=len(test_data), freq='D', include_history=False)
forecast = model.predict(future)

# # Plot the results
fig = model.plot(forecast)

fig2 = model.plot_components(forecast)

plt.show()
