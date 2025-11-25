import pandas as pd 

df = pd.read_csv("../data/stock_with_indicators/PDR_with_indicators.csv")
print(df.shape)
from sklearn.preprocessing import MinMaxScaler
