from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, RepeatVector, SpatialDropout1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from eval import evaluator
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import numpy as np
from numpy.random import seed
import random
import os
os.environ['PYTHONHASHSEED']=str(1)
random.seed(1)
seed(1)
tf.random.set_seed(1)
import math


class Learner(object):
    def __init__(self, name, max_len, vocab, load_model=False):
        if not load_model:
            self.model = self.build_model(name, length=max_len, vocab=vocab)
        else:
            self.model = False

    def build_model(self, name, length, vocab):
        if name == 'rcnn_splitted':
            model = self.rcnn_splitted(length, vocab)
        elif name == 'rcnn1':
            model = self.rcnn1(length, vocab)
        elif name == 'rcnn2':
            model = self.rcnn2(length, vocab)

        return model

    @staticmethod
    def step_decay(epoch):
        initial_lrate = 0.01
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    def input_layer(self, length, vocab):
        inputs1 = Input(shape=(length,))
        inputs2 = Input(shape=(length,))
        inputs3 = Input(shape=(length,))
        embedding1 = Embedding(len(vocab['words'].keys()) + 1, 200)(inputs1)
        embedding2 = Embedding(len(vocab['lemmas'].keys()) + 1, 200)(inputs2)
        embedding3 = Embedding(len(vocab['tags'].keys()) + 1, 15)(inputs3)
        merged = concatenate([embedding1, embedding2, embedding3])
        inputs = [inputs1, inputs2, inputs3]
        return merged, inputs

    def rcnn_splitted(self, length, vocab):
        # channel 1
        inputs1 = Input(shape=(length,))
        embedding1 = Embedding(len(vocab['words'].keys()) + 1, 1000)(inputs1)
        # batch_norm1 = BatchNormalization()(embedding1)
        conv1 = Conv1D(filters=64, kernel_size=4, activation='relu')(embedding1)
        drop1 = Dropout(0.2)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        # flat1 = Flatten()(pool1)
        lstm1 = LSTM(256, recurrent_dropout=0, activation='tanh', recurrent_activation='sigmoid')(pool1)
        # channel 2
        inputs2 = Input(shape=(length,))
        embedding2 = Embedding(len(vocab['lemmas'].keys()) + 1, 500)(inputs2)
        # batch_norm2 = BatchNormalization()(embedding2)
        conv2 = Conv1D(filters=64, kernel_size=4, activation='relu')(embedding2)
        drop2 = Dropout(0.2)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        # flat2 = Flatten()(pool2)
        lstm2 = LSTM(256, recurrent_dropout=0, activation='tanh')(pool2)
        # channel 3
        inputs3 = Input(shape=(length,))
        embedding3 = Embedding(len(vocab['tags'].keys()) + 1, 10)(inputs3)
        # batch_norm3 = BatchNormalization()(embedding3)
        conv3 = Conv1D(filters=64, kernel_size=4, activation='relu')(embedding3)
        drop3 = Dropout(0.2)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        # flat3 = Flatten()(pool3)
        lstm3 = LSTM(256, recurrent_dropout=0, activation='tanh', recurrent_activation='sigmoid')(pool3)
        # merge
        # merged = concatenate([flat1, flat2, flat3])
        merged = concatenate([lstm1, lstm2, lstm3])
        # interpretation
        dense1 = Dense(1000, activation='relu')(merged)
        outputs = Dense(5, activation='softmax')(dense1)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        # compile
        adam = Adam(learning_rate=0.001)
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=adam, metrics=['accuracy'])
        # summarize
        print(model.summary())
        return model

    def rcnn1(self, length, vocab):
        input_layer, inputs = self.input_layer(length, vocab)
        sp_drop = SpatialDropout1D(0.25)(input_layer)
        # drop1 = Dropout(0.25)(merged)
        conv1 = Conv1D(filters=256, kernel_size=8, activation='relu')(sp_drop)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        conv2 = Conv1D(filters=256, kernel_size=4, activation='relu')(pool1)
        pool2 = MaxPooling1D(pool_size=2)(conv2)
        conv3 = Conv1D(filters=256, kernel_size=2, activation='relu')(pool2)
        pool3 = MaxPooling1D(pool_size=2)(conv3)
        lstm1 = LSTM(256, return_sequences=True)(pool3)
        lstm2 = LSTM(256, recurrent_dropout=0.2)(lstm1)
        dense1 = Dense(1000, activation='relu')(lstm2)
        outputs = Dense(5, activation='softmax')(dense1)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])

        print(model.summary())

        return model

    def rcnn2(self, length, vocab):
        input_layer, inputs = self.input_layer(length, vocab)

        conv1 = Conv1D(filters=32, kernel_size=8, strides=2, activation='relu', padding='same')(input_layer)
        bn1 = BatchNormalization()(conv1)
        pool1 = MaxPooling1D(pool_size=2, strides=2, padding='same')(bn1)

        conv2 = Conv1D(filters=32, kernel_size=4, strides=2, activation='relu', padding='same')(pool1)
        bn2 = BatchNormalization()(conv2)
        pool2 = MaxPooling1D(pool_size=2, strides=2, padding='same')(bn2)

        conv3 = Conv1D(filters=32, kernel_size=4, strides=1, activation='relu', padding='same')(pool2)
        bn3 = BatchNormalization()(conv3)

        # Global Layers
        gmaxpl = GlobalMaxPooling1D()(bn3)
        gmeanpl = GlobalAveragePooling1D()(bn3)
        mergedlayer = concatenate([gmaxpl, gmeanpl], axis=1)

        fl = Flatten()(mergedlayer)
        rv = RepeatVector(300)(mergedlayer)
        lstm1 = LSTM(128, return_sequences=True)(bn3)
        do3 = Dropout(0.5)(lstm1)

        lstm2 = LSTM(64)(do3)
        do4 = Dropout(0.2)(lstm2)

        flat = Flatten()(mergedlayer)
        dense1 = Dense(100, activation='relu')(do4)
        outputs = Dense(5, activation='softmax')(mergedlayer)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])

        print(model.summary())

        return model

    def fit(self, X_train, Y_train, X_val, Y_val):
        self.model.fit([X_train['words'], X_train['lemmas'], X_train['tags']], Y_train,
                       validation_data=([X_val['words'], X_val['lemmas'], X_val['tags']], Y_val), epochs=20)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path)

    def predict(self, X_test, test_data, vocab, path):
        labels = {v: k for k, v in vocab['labels'].items()}
        y_pred = self.model.predict([X_test['words'], X_test['lemmas'], X_test['tags']])
        y_pred = np.argmax(y_pred, axis=1).tolist()
        with open(path, 'w') as outfile:
            for i, pred in enumerate(y_pred):
                print(test_data[i][0] + "|" + test_data[i][1] + "|" + test_data[i][2] + "|" + labels[pred],
                      file=outfile)

    def evaluate(self, files_path, results_path):
        evaluator.evaluate("DDI", files_path, results_path)

    def plot_model(self, path):
        plot_model(self.model, to_file=path, show_shapes=True)