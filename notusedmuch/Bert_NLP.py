import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

filename = r'C:\Users\u65988\Downloads\macbeth.txt'

text = (open(filename).read()).lower()

unique_chars = sorted(list(set(text)))

char_to_int = {}
int_to_char = {}

for i, c in enumerate (unique_chars):
    char_to_int.update({c: i})
    int_to_char.update({i: c})
    
# preparing input and output dataset
X = []
Y = []

for i in range(0, len(text) - 50, 1):
    sequence = text[i:i + 50]
    label =text[i + 50]
    X.append([char_to_int[char] for char in sequence])
    Y.append(char_to_int[label])

# reshaping, normalizing and one hot encoding
X_modified = numpy.reshape(X, (len(X), 50, 1))
X_modified = X_modified / float(len(unique_chars))
Y_modified = np_utils.to_categorical(Y)
