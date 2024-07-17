import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM  #long short term Memory

# load Data
company = 'FB'
start = dt.datetime(2020, 1, 1)  #debut
end = dt.datetime.now()

data = web.DataReader(company, 'yahoo', start, end)  #  données de entreprise moteur de recherche début fin

# prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

predictions_days = 60
x_train = []
y_train = []

for x in range(predictions_days, len(scaled_data)):
    x_train.append(scaled_data[x - predictions_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#build the model

model = Sequential()  #création d'un model sequentiel

model.add(LSTM(units=75, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=75, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=75))
model.add(Dropout(0.2))
model.add(LSTM(units=1))  #Prediciton du prochain prix

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

''' Tester la précision du model via des données existantes'''

test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()
test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']))

model_inputs = total_dataset[len(total_dataset) - len(test_data) - predictions_days:].values
model_inputs = model_inputs.reshapea(-1, 1)
model_inputs = scaler.transform(model_inputs)

#Faire des prédictions

x_test = []
for x in range(predictions_days, len(model_inputs)):
    x_test.append(model_inputs[x - predictions_days:x, 0])

x_test = np.array(x_test)
x_test = x_test.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

#Plot de the test prediction

plt.plot(actual_prices, color="blue", label="Actual")
plt.plot(predicted_prices, color="red", label="Predicted")
plt.title(f"{company}Actual vs Predicted")
plt.xlabel("Time")
plt.ylabel(f"{company}Price")
plt.legend()
plt.show()
