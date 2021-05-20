from src.pra4 import FeatureExtractor, Model, DataLoader, ModelBinary
import platform
from config import config_file


resources_section = f"Resources-{platform.system()}"
resources_config = config_file.get(resources_section)
data_config = config_file.get(f"PRA4-Data-{platform.system()}")

extract_features = False
train = True
test = True

if __name__ == "__main__":
    train_features_path = f"{data_config.output}train_features.out"
    vocab_path = f"{data_config.output}vocabulary.out"
    test_features_path = f"{data_config.output}test_features.out"
    devel_features_path = f"{data_config.output}devel_features.out"

    train_feature_extractor = FeatureExtractor(path=data_config.train, out_path=train_features_path, vocab_path=vocab_path)
    devel_feature_extractor = FeatureExtractor(path=data_config.devel, out_path=devel_features_path)
    test_feature_extractor = FeatureExtractor(path=data_config.test, out_path=test_features_path)

    if extract_features:
        train_feature_extractor.get_features(vocab=True)
        test_feature_extractor.get_features()
        devel_feature_extractor.get_features()

    data_loader_train = DataLoader(path=train_features_path, batch_size=128, vocabuary_path=vocab_path, make_vocab=True)
    data_loader_devel = DataLoader(path=devel_features_path, batch_size=1000, vocabuary_path=vocab_path)
    data_loader_test = DataLoader(path=test_features_path, batch_size=128, vocabuary_path=vocab_path, binary_path=None)

    print('Start training...')
    if train:


        data_loader_train.train_encoder()
        data_loader_devel.train_encoder()
        data_loader_test.train_encoder()

        binary_model = ModelBinary()
        binary_model.train(data_loader_train)

        predictions = binary_model.predict(data_loader_test)

        model = Model()
        model.train(data_loader_train, data_loader_devel)

        model.evaluate(files_path=data_config.train, results_path='out/pra4/results_train.out')


    if test:

        #result_binary_path = f"{data_config.output}result_binary.out"
        #binary_model.predict_binary(data_loader_test, result_binary_path)

        results_path = f"{data_config.output}result.out"

        model.predict(data_loader_test, results_path, predictions)

        model.evaluate(files_path=data_config.test, results_path=results_path)



