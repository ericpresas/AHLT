from src.pra5 import DataLoader
import platform
from config import config_file
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical


resources_section = f"Resources-{platform.system()}"
resources_config = config_file.get(resources_section)
data_config = config_file.get(f"PRA5-Data-{platform.system()}")

if __name__ == "__main__":
    vocab_file = f"{data_config.output}vocab.pkl"

    model_file = f"{data_config.output}model.tf"

    pred_file = f"{data_config.output}predictions.txt"

    with open(vocab_file, 'rb') as fp:
        vocab = pickle.load(fp)

    data_loader = DataLoader()

    test_data = data_loader.load_data(data_config.test)

    X_test = data_loader.encode_words(test_data, vocab)

    with open(data_config.output+'X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)

    with open(data_config.output+'dataset_test.pkl', 'wb') as f:
        pickle.dump(test_data, f)