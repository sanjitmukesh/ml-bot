import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df = df.replace({'label': {'ham': 0, 'spam': 1}})

# feature (input): message
X = df['message']

# target (output): label
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=15
)

df['word_count'] = df['message'].apply(lambda x: len(x.split()))
average_length = int(df['word_count'].mean())

model = keras.Sequential()

vectorizer = TextVectorization (
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=average_length
)
vectorizer.adapt(df['message'])
model.add(vectorizer)

embedding_layer = Embedding(
    input_dim=10000,
    output_dim=64,
)
model.add(embedding_layer)

pooling_layer = GlobalAveragePooling1D()
model.add(pooling_layer)

hidden_layer = Dense (
    units=64,

    # introduces nonlinearity so the model can learn complex patterns
    activation='relu'
)
model.add(hidden_layer)

output_layer = Dense (
    # single neuron output; predicts probability of spam (1) or ham (0)
    units=1,

    # squashes neuron output between 0 and 1 - used for binary classification
    activation='sigmoid'
)
model.add(output_layer)

model.compile (
    # optimizer decides how to adjust weights so predictions improve based on the loss
    # moves weights in the direction that most reduces the lost on the next round
    optimizer='adam', 

    # loss function measures how wrong the model's prediction is
    # gives optimizer a number to minimize: big loss = very wrong, small loss = close to correct
    loss='binary_crossentropy', 

    # for monitoring training performance
    # accuracy -> what % of predictions are correct
    metrics=['accuracy']
)

# train -> evaluate -> predict

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

model.fit (
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
    epochs=10
)

model.evaluate (
    x=X_test,
    y=y_test
)

model.save("spam_classifier.keras")