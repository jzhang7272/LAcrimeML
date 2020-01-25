import pandas as pd
import numpy as np
import datetime

# Categories analyzed
columns = ['Arrest Date', 'Time', 'Charge Group Description', 'Location']

df = pd.read_csv("arrests.csv", converters={'Time': lambda x: str(x)}, delimiter=',')
df1 = pd.DataFrame(df, columns=columns)
print(df.shape)

# Remove duplicates
print(df1.duplicated().sum())
df1.drop_duplicates(keep='first', inplace=True)
print(df1.duplicated().sum())

# Remove rows with no data
print(df1.isna().sum())
df1.dropna(inplace=True)
print(df1.isna().sum())

# Splitting to x/y coord
data = df1['Location'].str.split('\'', n=-1, expand=True)
df1['X'] = data[3]
df1['Y'] = data[11]

# Date to day of week
df2 = pd.read_csv("clean.csv")
df2['Date'] = pd.to_datetime(df2['Date'])
df2['Date'] = df2['Date'].dt.dayofweek
df1.insert(1, 'Day of Week', df2['Date'])

# Splitting to hours and minutes
df1.Time = df1.Time.astype(str)
df1.Time = df1.Time.str[:2] + ':' + df1.Time.str[-2:]
time = df1['Time'].str.split(":", n=1, expand=True)
df1['Hour'] = time[0]
df1['Minute'] = time[1]
df1['Hour'] = pd.to_numeric(df1['Hour'])
df1['Minute'] = pd.to_numeric(df1['Minute'])
df1.drop(columns=['Time'], inplace=True)

category = df1['Charge Group Description']
df1.insert(0, 'Category', category)
df1['Category'] = df1.Category.astype('category')
print(df1['Category'].dtypes)
df1.drop(columns='Charge Group Description', inplace=True)
df1.drop(columns='Location', inplace=True)
df1.drop(columns='Arrest Date', inplace=True)
# df1.drop(columns='Date', inplace=True)

df1.to_csv("clean.csv")

# data = df1['Arrest Date'].str.split('T', n=-1, expand=True)
# df1.insert(1, 'Day of Week', data[0])
# print(df1['Day of Week'])
# df1['Day of Week'] = pd.to_datetime(df1['Day of Week'])
# df1['Day of Week'] = df1['Day of Week'].dt.dayofweek.astype(float)
