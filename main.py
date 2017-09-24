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

batch_size = 500

train_variant = pd.read_csv("./inputs/training_variants")
train_text = pd.read_csv("./inputs/training_text", sep="\|\|", engine='python', header=None, skiprows=1,
                         names=["ID", "Text"])
train = pd.merge(train_variant, train_text, how='left', on='ID')
train = train.sample(1000)

def text_clean(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = text.strip().lower().split()
    stops = stopwords.words('english')
    text = [w for w in text if not w in stops]
    text = [w for w in text if not w in string.punctuation]
    return " ".join(text)


train['Text'] = train['Text'].apply(text_clean)

max_document_length = 10000
embedding_size = 100

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(train['Text'])))

vocab_size = len(vocab_processor.vocabulary_)
word_dictionary_rev = sorted(vocab_processor.vocabulary_._mapping.items(), key = lambda x : x[1])
valid_words = [word_dictionary_rev[i][0] for i in np.random.choice(len(word_dictionary_rev),4)]

# shuffle_indices = np.random.permutation(np.arange(len(x)))
# x_shuffled = x[shuffle_indices]
# y_shuffled = train['Class'].values[shuffle_indices]
word_dictionary = vocab_processor.vocabulary_._mapping
valid_examples = [word_dictionary[i] for i in valid_words]

def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        idx = np.random.choice(len(sentences))
        rand_sentence = sentences[idx]
        window_sequences = [rand_sentence[max((ix - window_size), 0):(ix + window_size + 1)].tolist() for ix, x in
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


embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))

x_inputs = tf.placeholder(tf.int32, shape=[batch_size])
y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

embed = tf.nn.embedding_lookup(embeddings, x_inputs)

nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocab_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                     biases=nce_biases,
                                     labels=y_target,
                                     inputs=embed,
                                     num_sampled=batch_size / 2,
                                     num_classes=vocab_size))

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
init = tf.initialize_all_variables()

with tf.Session() as sess:

    sess.run(init)

    loss_vec = []
    loss_x_vec = []
    for i in range(50000):
        batch_inputs, batch_labels = generate_batch_data(x, batch_size, 2)
        feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}
        # Run the train step
        sess.run(optimizer, feed_dict=feed_dict)
        # Return the loss
        if (i + 1) % 500 == 0:
            loss_val = sess.run(loss, feed_dict=feed_dict)
            loss_vec.append(loss_val)
            loss_x_vec.append(i + 1)
            print("Loss at step {} : {}".format(i + 1, loss_val))

        # Validation: Print some random words and top 5 related words
        if (i + 1) % 500 == 0:
            sim = sess.run(similarity, feed_dict=feed_dict)
            for j in range(len(valid_words)):
                try:
                    valid_word = word_dictionary_rev[valid_examples[j]]
                    top_k = 5  # number of nearest neighbors
                    nearest = (-sim[j, :]).argsort()[1:top_k + 1]
                    log_str = "Nearest to {}:".format(valid_word)
                    for k in range(top_k):
                        close_word = word_dictionary_rev[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
                except KeyError:
                    print('nothing for {}'.format(valid_examples[j]))
