import pandas as pd 
import quandl
import math
import numpy as np 
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
df = quandl.get('WIKI/GOOGL')
#print(df.head)

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

#print(df.head)

df['Hl_pct'] = (df['Adj. High'] - df['Adj. Low'])/ df['Adj. Low'] * 100
df['pct_c'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] * 100

df = df[['Hl_pct','pct_c','Adj. Close','Adj. Volume']]
#print(df.head)

forcast_col = 'Adj. Close'

df.fillna(-9999, inplace = True)

forcast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forcast_col].shift(-forcast_out)
df.dropna(inplace=True)
print(df.head)
#print(df.tail)