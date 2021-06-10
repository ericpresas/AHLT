from src.pra6 import Learner, DataLoader
import platform
from config import config_file
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical


resources_section = f"Resources-{platform.system()}"
resources_config = config_file.get(resources_section)
data_config = config_file.get(f"PRA6-Data-{platform.system()}")

if __name__ == "__main__":
    vocab_file = f"{data_config.output}vocab.pkl"

    model_file = f"{data_config.output}model2_100.h5"

    pred_file = f"{data_config.output}predictions.txt"

    with open(vocab_file, 'rb') as fp:
        vocab = pickle.load(fp)

    data_loader = DataLoader()

    test_data = data_loader.load_data(data_config.test)

    X_test = data_loader.encode_words(test_data, vocab)
    X_test['words'] = np.array(X_test['words'])
    X_test['lemmas'] = np.array(X_test['lemmas'])
    X_test['tags'] = np.array(X_test['tags'])

    Y_test = data_loader.encode_labels(test_data, vocab)

    y_test = np.array([label[0] for label in Y_test])
    y_test = to_categorical(y_test)

    learner = Learner(name='rcnn', max_len=100, vocab=vocab)
    learner.load_model(model_file)

    learner.predict(X_test, test_data, vocab, pred_file)

    learner.evaluate(data_config.test, pred_file)
    print('a')