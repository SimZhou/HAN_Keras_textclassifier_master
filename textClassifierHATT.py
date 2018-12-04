# author - Richard Liao
# Dec 26 2016
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re

# from bs4 import BeautifulSoup

import sys
import os

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed#, Merge
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# GLOVE_DIR = "D:\\Yihua\\_myDeepLearning\\Datasets\\GloVe"
# IMDB_DIR = 'D:\\Yihua\\_myDeepLearning\\Datasets\\IMDB'

GLOVE_DIR = "/glove"
IMDB_DIR = '/imdb'

# Define a function to clean strings
def clean_str(string):	
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased, "\", "'", '"' are deleted
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


data_train = pd.read_csv(os.path.join(IMDB_DIR, 'labeledTrainData.tsv'), sep='\t')  # Loading training data
print (data_train.shape)    # Print the shape of training data

import nltk
from nltk import tokenize
nltk.download('punkt')

reviews_sentences = []    # list of sentence lists
labels = []     # labels
reviews = []      # list of reviews

for idx in range(data_train.review.shape[0]):   # for the length of training samples
    # text = BeautifulSoup(data_train.review[idx], 'lxml')    # read in reviews to BS4
    text = clean_str(data_train.review[idx])     # encode review text to ascii and ignore un-encodable characters
    reviews.append(text)  # list of review posts
    sentences = tokenize.sent_tokenize(text)    # list of sents within this single review
    reviews_sentences.append(sentences)   # list of review posts with lists of sentences
    labels.append(data_train.sentiment[idx])    # list of labels

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)    # 
tokenizer.fit_on_texts(reviews)     # list of texts to train on

data = np.zeros((len(reviews), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')    # zero matrix with dim 3: num of texts, max num of sents, max num of words in sents

# getting data_x matrices
for i, sentences in enumerate(reviews_sentences):     
    for j, sent in enumerate(sentences): 
        if j < MAX_SENTS:       # only deal with 'MAX_SENTS' of sentences for each post
            wordTokens = text_to_word_sequence(sent)    # Word tokenization
            k = 0      # 
            for _, word in enumerate(wordTokens): 
                if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS: 
                    data[i, j, k] = tokenizer.word_index[word] 
                    k = k + 1 

# getting word indices
word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))

# getting data_y matrices
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# Randomizing the dataset x & y
indices = np.arange(data.shape[0])
np.random.shuffle(indices)          # Randomize the dataset
data = data[indices]                # Randomize the dataset
labels = labels[indices]            # Randomize the dataset
# Spliting test set
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Number of positive and negative reviews in training and validation set')
print (y_train.sum(axis=0))
print (y_val.sum(axis=0))


# Loading Word Embeddings, into a dictionary
GLOVE_vec = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding = 'utf-8')
for line in f:
    values = line.split()
    word = values[0]
    word_vector = np.asarray(values[1:], dtype='float32')
    GLOVE_vec[word] = word_vector
f.close()

print('Total %s word vectors.' % len(GLOVE_vec))


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, index in word_index.items():
    _vector = GLOVE_vec.get(word)
    if _vector is not None:
        # words not found in GLOVE_vec will be all-zeros.
        embedding_matrix[index] = _vector





# building Hierachical Attention network

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True,
                            mask_zero=True)


# class AttLayer(Layer):
#     def __init__(self, attention_dim):
#         self.init = initializers.get('normal')
#         self.supports_masking = True
#         self.attention_dim = attention_dim
#         super(AttLayer, self).__init__()
# #
#     def build(self, input_shape):
#         assert len(input_shape) == 3
#         self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
#         self.b = K.variable(self.init((self.attention_dim, )))
#         self.u = K.variable(self.init((self.attention_dim, 1)))
#         self.trainable_weights = [self.W, self.b, self.u]
#         super(AttLayer, self).build(input_shape)
# #
#     def compute_mask(self, inputs, mask=None):
#         return mask
# #
#     def call(self, x, mask=None):
#         # size of x :[batch_size, sel_len, attention_dim]
#         # size of u :[batch_size, attention_dim]
#         # uit = tanh(xW+b)
#         uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
#         ait = K.dot(uit, self.u)
#         ait = K.squeeze(ait, -1)
# #
#         ait = K.exp(ait)
# #
#         if mask is not None:
#             # Cast the mask to floatX to avoid float64 upcasting in theano
#             ait *= K.cast(mask, K.floatx())
#         ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
#         ait = K.expand_dims(ait)
#         weighted_input = x * ait
#         output = K.sum(weighted_input, axis=1)
# #
#         return output
# #
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], input_shape[-1])

class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()
#
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)
#
    def compute_mask(self, inputs, mask=None):
        # return mask
        return None
#
    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
#
        ait = K.exp(ait)
#
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
#
        return output
#
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_att = AttLayer(100)(l_lstm)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_att_sent = AttLayer(100)(l_lstm_sent)
preds = Dense(2, activation='softmax')(l_att_sent)                              
# 终极步骤，构建模型，输入为 review_input, 输出为 preds
model = Model(review_input, preds)                                              

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Hierachical attention network")
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=50)
