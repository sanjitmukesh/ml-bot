import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df = df.replace({'v1': {'ham': 0, 'spam': 1}})

print(df.head())