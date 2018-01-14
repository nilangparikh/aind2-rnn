import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for i in range(len(series)):
        if (((len(series) - 1) - i) - window_size >= 0):
            X.append(series[i:window_size+i])

    y = series[window_size:]

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))

    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', '?']

    remove_chars = {'@', '^', '--', '-', ';', '{', '}', '+', '-', '<', '>', '\"', '#', '*', '=', '&', '(', ')', '\'',
                    '\\', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '$', '%', '~', '_', '[', ']', '|', '`',
                    ':', ' , ', ' ., ', ' .f. . ', ' .f. ', ' .e. . ', '\n', '\r'}

    for ch in remove_chars:
        if ch in text:
            text = text.replace(ch, ' ')

    text = text.replace(' .', '.')
    text = text.replace(' s', 's')
    text = text.replace('  ', ' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    steps = ((len(text) - window_size) + 1) // step_size
    inputs = [text[step * step_size:step * step_size + window_size] for step in range(steps)]
    outputs = [text[step * step_size + window_size] for step in range(steps)]

    #n = 0
    #text_list = list(text)
    #text_len = len(text_list)

    #while n + window_size < text_len:
    #    inputs.append(text_list[n:(n+window_size)])
    #    outputs.append(text_list[n+window_size])
    #    n += step_size

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='linear'))
    model.add(Activation('softmax'))

    return model