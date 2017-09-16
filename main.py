import os
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
import tensorflow as tf

batch_size = 50

train_variant = pd.read_csv("./inputs/training_variants")
train_text = pd.read_csv("./inputs/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train = pd.merge(train_variant, train_text, how='left', on='ID')
train = train.sample(10)

def text_clean(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = text.strip().lower().split()
    stops = stopwords.words('english')
    text = [w for w in text if not w in stops]
    return " ".join(text)

train['Text'] = train['Text'].apply(text_clean)


vocab_processor = learn.preprocessing.VocabularyProcessor(10)
x = np.array(list(vocab_processor.fit_transform(train['Text'])))

y = train['Class'].values

shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

X_train, X_test, y_train, y_test = train_test_split(x_shuffled, y_shuffled, test_size=0.33)


train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

