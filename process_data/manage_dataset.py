import pandas as pd
import numpy as np

df = pd.read_csv('merged_news_EA.csv')
# print(df.head())
print(df.shape[0])
nanrows = df[df['Close_Price'].isna()]
# print(nanrows)

df = df.dropna()

is_NaN = df.isnull()
row_has_NaN = is_NaN.any(axis=1).sum()
print(row_has_NaN)
print(df.shape[0])

df.rename(columns={"score": "news"}, inplace = True)
print(df.head())

df.to_csv("EA_trimmed.csv", index=False)