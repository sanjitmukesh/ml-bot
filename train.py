import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import TextVectorization

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df = df.replace({'v1': {'ham': 0, 'spam': 1}})

# feature (input): message
# target (output): label
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=15
)

df['word_count'] = df['message'].apply(lambda x: len(x.split()))
average_length = int(df['word_count'].mean())

vectorizer = TextVectorization (
    max_tokens=10000,
    output_mode="int",
    output_sequence_length=average_length
)

vectorizer.adapt(df['message'])