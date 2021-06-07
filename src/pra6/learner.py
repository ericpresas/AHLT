from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import numpy as np
from eval import evaluator


class Learner(object):
    def __init__(self, max_len, vocab_size):
        self.model = self.define_model(length=max_len, vocab_size=vocab_size)

    @staticmethod
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    @staticmethod
    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def define_model(self, length, vocab_size):
        # channel 1
        inputs1 = Input(shape=(length,))
        embedding1 = Embedding(vocab_size, 100)(inputs1)
        #batch_norm1 = BatchNormalization()(embedding1)
        conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)
        # channel 2
        inputs2 = Input(shape=(length,))
        embedding2 = Embedding(vocab_size, 100)(inputs2)
        #batch_norm2 = BatchNormalization()(embedding2)
        conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)
        # channel 3
        inputs3 = Input(shape=(length,))
        embedding3 = Embedding(vocab_size, 100)(inputs3)
        #batch_norm3 = BatchNormalization()(embedding3)
        conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)
        # merge
        merged = concatenate([flat1, flat2, flat3])
        # interpretation
        dense1 = Dense(10, activation='relu')(merged)
        outputs = Dense(5, activation='softmax')(dense1)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        # compile
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
        # summarize
        print(model.summary())
        return model

    def fit(self, X_train, Y_train, X_val, Y_val):
        self.model.fit([X_train, X_train, X_train], Y_train, validation_data=([X_val, X_val, X_val], Y_val), epochs=10)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path)

    def predict(self, X_test, test_data, vocab, path):
        labels = {v: k for k, v in vocab['labels'].items()}
        y_pred = self.model.predict([X_test, X_test, X_test])
        y_pred = np.argmax(y_pred, axis=1).tolist()
        with open(path, 'w') as outfile:
            for i, pred in enumerate(y_pred):
                print(test_data[i][0] + "|" + test_data[i][1] + "|" + test_data[i][2] + "|" + labels[pred], file=outfile)

    def evaluate(self, files_path, results_path):
        evaluator.evaluate("DDI", files_path, results_path)