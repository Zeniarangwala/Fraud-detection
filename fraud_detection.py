import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)


from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)


from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)
customers = dataset.iloc[:, 1:].values

# Creating the dependent variable
# Initialize vector of zeros
is_fraud = np.zeros(len(dataset))
# If a customer ID is found in list of frauds, change its value from 0 to 1
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


# Creating an ANN to determine how likely a customer is to have committed fraud

# Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


classifier = Sequential()

classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu', input_dim = 15))
# Droput regularization for potential overfitting
classifier.add(Dropout(0.1))

classifier.add(Dropout(0.1))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 

classifier.fit(customers, is_fraud, batch_size = 1, nb_epoch = 2)

# Predict each probabilty of each customer of having committed fraud
y_pred = classifier.predict(customers)

# Create 2-d array of of customer IDs and predicted probabilites
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)


# Sorting customers from least to most likely of having committed fraud

y_pred = y_pred[y_pred[:, 1].argsort()]
