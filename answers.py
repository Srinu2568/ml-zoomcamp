import numpy as np
import pandas as pd
import csv

# problem1
# Version of numpy = 1.21.1

# problem2
# Version of pandas = 1.3.1

# Reading data from csv and Framing the data using read_csv method
df = pd.read_csv("data.csv")

# problem3
# print(df.groupby("Make").MSRP.mean())

# problem4
# print(((df['Engine HP'].isnull()) & df['Year'] >= 2015).sum())

# problem5
# mean_hp_before = df["Engine HP"].mean().round()
# print(mean_hp_before)
# mean_hp_after = df["Engine HP"].fillna(0).mean().round()
# print(mean_hp_after)

# problem6
# x = df.loc[df["Make"] == "Rolls-Royce"]
# y = x[["Engine HP", "Engine Cylinders", "highway MPG"]]
# y = y.drop_duplicates()
# X = np.array(y)
# T = np.transpose(X)
# XTX = T.dot(X)
# XTX = np.linalg.inv(XTX)
# sum = XTX.sum()
# print(sum)

# problem7
Y = np.array([1000, 1100, 900, 1200, 1000, 850, 1300])
x = df.loc[df["Make"] == "Rolls-Royce"]
y = x[["Engine HP", "Engine Cylinders", "highway MPG"]]
y = y.drop_duplicates()
X = np.array(y)
T = np.transpose(X)
XTX = T.dot(X)
XTX = np.linalg.inv(XTX)
result = np.dot(XTX, np.transpose(X))
w = np.dot(result, Y)
print(w[0])
