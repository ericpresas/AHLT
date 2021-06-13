from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate
from keras_crf import CRFModel
from tensorflow.keras.optimizers import RMSprop, Adam
from keras import losses
from keras.backend import tf
from keras import regularizers
from keras import models
from keras.models import load_model
import keras


import os


class Learner(object):
    def __init__(self):
        pass

    def build_network_affixes(self, idx):
        """
        Create network for the learner
        """

        # sizes
        n_words = len(idx['words'])
        n_pref = len(idx['pref'])
        n_suff = len(idx['suf'])
        n_labels = len(idx['labels'])
        max_len = idx['maxlen']

        print(n_words)

        # Inputs
        word_inp = Input(shape=(max_len,))
        pref_inp = Input(shape=(max_len,))
        suff_inp = Input(shape=(max_len,))

        # Embedding layers
        emb_word = Embedding(input_dim=n_words + 2, output_dim=100,
                             input_length=max_len, mask_zero=True)(word_inp)

        emb_pref = Embedding(input_dim=n_pref + 2, output_dim=100,
                             input_length=max_len, mask_zero=True)(pref_inp)

        emb_suff = Embedding(input_dim=n_suff + 2, output_dim=100,
                             input_length=max_len, mask_zero=True)(suff_inp)

        X = concatenate([emb_word, emb_pref, emb_suff])

        # Bidirectional LSTM
        main_biLSTM = Bidirectional(LSTM(units=50, return_sequences=True,
                                         recurrent_regularizer=regularizers.L1(0.1),
                                         recurrent_dropout=0.5))(X)

        out = TimeDistributed(Dense(50, activation='relu'))(main_biLSTM)

        base = Model(inputs=[word_inp, pref_inp, suff_inp], outputs=out)

        # CRF
        model = CRFModel(base, n_labels)

        opt = RMSprop(learning_rate=1e-3)

        model.compile(
            optimizer=opt,
            metrics=['acc']
        )
        return model

    def load_model(self, model_path):
        model = load_model(model_path)
        return model

    def predict(self, model_path, dataset, X_test, out_path):

        # Reads the features from 'features_path' and get the tags (labels)
        with open(out_path, 'w') as outfile:
            model = self.load_model(model_path)

            preds = model.predict([X_test['words'], X_test['pref'], X_test['suf']])
            print('a')
