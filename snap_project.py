import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


dataset = pd.read_csv("SNAP_history_1969_2019.csv")
dataset.replace(',', '', regex=True, inplace=True)

x = dataset.iloc[:, 3:4].astype(float).values
y = dataset.iloc[:, -1].astype(float).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

plt.scatter(x, y, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Total Benefit vs. Total Cost ( Linear Regression )")
plt.xlabel("Total Benefit")
plt.ylabel("Total Cost")
plt.show()

plt.scatter(x_test, y_pred, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Total Benefit vs. Total Cost ( Linear Regression )")
plt.xlabel("Total Benefit")
plt.ylabel("Total Cost")
plt.show()
