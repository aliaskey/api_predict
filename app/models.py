# Ce fichier est prévu pour les modèles de deep learning (comme LSTM).
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

def create_lstm_model(vocab_size, input_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=input_length))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
