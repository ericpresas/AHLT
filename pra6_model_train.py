from src.pra6 import Learner
import platform
from config import config_file
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical


resources_section = f"Resources-{platform.system()}"
resources_config = config_file.get(resources_section)
data_config = config_file.get(f"PRA6-Data-{platform.system()}")

if __name__ == "__main__":
    X_train_file = f"{data_config.output}X_train.pkl"
    X_val_file = f"{data_config.output}X_val.pkl"
    Y_train_file = f"{data_config.output}Y_train.pkl"
    Y_val_file = f"{data_config.output}Y_val.pkl"
    vocab_file = f"{data_config.output}vocab.pkl"

    model_file = f"{data_config.output}model.h5"

    with open(vocab_file, 'rb') as fp:
        vocab = pickle.load(fp)

    with open(X_train_file, 'rb') as fp:
        X_train = np.array(pickle.load(fp))

    with open(Y_train_file, 'rb') as fp:
        Y_train = pickle.load(fp)

    with open(X_val_file, 'rb') as fp:
        X_val = np.array(pickle.load(fp))

    with open(Y_val_file, 'rb') as fp:
        Y_val = pickle.load(fp)

    y_train = np.array([label[0] for label in Y_train])
    y_val = np.array([label[0] for label in Y_val])

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    learner = Learner(max_len=100, vocab_size=len(vocab['words'].keys()) + 1)
    learner.fit(X_train, y_train, X_val, y_val)
    learner.save_model(model_file)
    print('a')