import os
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from matplotlib import pyplot as plt
import string

batch_size = 50

train_variant = pd.read_csv("./inputs/training_variants")
train_text = pd.read_csv("./inputs/training_text", sep="\|\|", engine='python', header=None, skiprows=1,
                         names=["ID", "Text"])
train = pd.merge(train_variant, train_text, how='left', on='ID')
train = train


def text_clean(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = text.strip().lower().split()
    stops = stopwords.words('english')
    text = [w for w in text if not w in stops]
    text = [w for w in text if not w in string.punctuation]
    return " ".join(text)


train['Text'] = train['Text'].apply(text_clean)

vocab_size = 10000

vocab_processor = learn.preprocessing.VocabularyProcessor(vocab_size)


def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        rand_sentence = np.random.choice(sentences)
        window_sequences = [rand_sentence[max((ix - window_size), 0):(ix + window_size + 1)] for ix, x in
                            enumerate(rand_sentence)]
        label_indices = [ix if ix < window_size else window_size for ix, x in enumerate(window_sequences)]

        if method == 'skip_gram':
            batch_and_labels = [(x[y], x[:y] + x[(y + 1):]) for x, y in zip(window_sequences, label_indices)]
            tuple_data = [(x, y_) for x, y in batch_and_labels for y_ in y]
        else:
            raise ValueError('Method {} not implmented yet.'.format(method))

        batch, labels = [list(x) for x in zip(*tuple_data)]
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])


    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]  # Convert to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))

    return (batch_data, label_data)
