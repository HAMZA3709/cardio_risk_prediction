import pandas as pd

df = pd.read_csv("data/cardio_train.csv", sep=';')
print(df.head())
print(df.shape)

