from src.pra5 import DataLoader
import platform
from config import config_file
import pickle


resources_section = f"Resources-{platform.system()}"
resources_config = config_file.get(resources_section)
data_config = config_file.get(f"PRA5-Data-{platform.system()}")

if __name__ == "__main__":
    X_train_file = f"{data_config.output}X_train.pkl"
    X_val_file = f"{data_config.output}X_val.pkl"
    Y_train_file = f"{data_config.output}Y_train.pkl"
    Y_val_file = f"{data_config.output}Y_val.pkl"
    vocab_file = f"{data_config.output}vocab.pkl"

    data_loader = DataLoader()
    train_data = data_loader.load_data(data_config.train)
    val_data = data_loader.load_data(data_config.devel)
    indexes_data = data_loader.create_indexs(train_data, max_length=100)

    X_train = data_loader.encode_words(train_data, indexes_data)
    Y_train = data_loader.encode_labels(train_data, indexes_data)

    X_val = data_loader.encode_words(val_data, indexes_data)
    Y_val = data_loader.encode_labels(val_data, indexes_data)



    with open(X_train_file, 'wb') as f:
        pickle.dump(X_train, f)

    with open(Y_train_file, 'wb') as f:
        pickle.dump(Y_train, f)

    with open(X_val_file, 'wb') as f:
        pickle.dump(X_val, f)

    with open(Y_val_file, 'wb') as f:
        pickle.dump(Y_val, f)

    with open(vocab_file, 'wb') as f:
        pickle.dump(indexes_data, f)

