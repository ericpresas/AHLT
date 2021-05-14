from src.pra4 import FeatureExtractor, Model, DataLoader
import platform
from config import config_file
from sklearn.metrics import f1_score

resources_section = f"Resources-{platform.system()}"
resources_config = config_file.get(resources_section)
data_config = config_file.get(f"PRA4-Data-{platform.system()}")

extract_features = False
train = True

if __name__ == "__main__":
    train_features_path = f"{data_config.output}train_features.out"
    vocab_path = f"{data_config.output}vocabulary.out"
    test_features_path = f"{data_config.output}test_features.out"

    train_feature_extractor = FeatureExtractor(path=data_config.train, out_path=train_features_path, vocab_path=vocab_path)
    test_feature_extractor = FeatureExtractor(path=data_config.test, out_path=test_features_path)
    if extract_features:
        train_feature_extractor.get_features(vocab=True)
        test_feature_extractor.get_features()

    data_loader_train = DataLoader(path=train_features_path, batch_size=64, vocabuary_path=vocab_path)
    #train_features, train_labels = train_feature_extractor.load_features()

    if train:
        model = Model()
        model.train(data_loader_train)

        train_features = []
        train_labels = []

        data_loader_test = DataLoader(path=test_features_path, batch_size=64, vocabuary_path=vocab_path)
        # TODO: save results in file
        for iteration in range(data_loader_test.num_batches):
            test_features, test_labels, info_batch = data_loader_test.load_features(iteration)
            pred_labels = model.predict(test_features)
            metric = f1_score(test_labels, list(pred_labels), average='macro')
            print(f"Iteration: {iteration}, score: {metric}")

    print('a')

