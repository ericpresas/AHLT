from src.pra2 import Parser
from src.pra2 import Learner
import platform
from config import config_file

resources_section = f"Resources-{platform.system()}"
resources_config = config_file.get(resources_section)
data_config = config_file.get(f"PRA2-Data-{platform.system()}")

if __name__ == "__main__":
    out_path = f"{data_config.output}result.out"

    parser = Parser(data_paths=data_config, resources_paths=resources_config, out_path=out_path)
    learner = Learner()

    extract_features = False

    if extract_features:
        print('Extract features')
        # Get Train and test features
        parser.path_features(output_file=f"{data_config.output}all_features_train.data", type_='train')
        parser.path_features(output_file=f"{data_config.output}all_features_test.data", type_='test')
        parser.path_features(output_file=f"{data_config.output}all_features_devel.data", type_='devel')

    estimate_params = False

    # Para usar el estimador downgrade scikit : pip install -U 'scikit-learn<0.24'
    if estimate_params:

        labels = ['B-drug', 'I-drug', 'B-drug_n', 'I-drug_n', 'B-group', 'I-group', 'B-brand', 'I-brand']
        best_params = learner.estimate_best_params(features_path=f"{data_config.output}all_features_devel.data", labels=labels)

    else:
        #best_params = {'algorithm': 'lbfgs', 'c1': 0.01, 'c2': 0.1, 'delta': 1e-05, 'linesearch': 'MoreThuente'}
        best_params = {'algorithm': 'lbfgs', 'c1': 0.00394884905617865, 'c2': 0.10734123931366764, 'delta': 7.507345527782286e-06, 'linesearch': 'StrongBacktracking'}
        #best_params = {'algorithm': 'lbfgs', 'c1': 0.26698808800350887, 'c2': 0.0035781435343593634, 'delta': 8.630770732257361e-05, 'linesearch': 'Backtracking'}

    print('Train')
    # Train the model
    learner.learn(features_path=f"{data_config.output}all_features_train.data", out_path=f"{data_config.output}model_best_all_features.crfsuite", params=best_params)

    predict = True

    if predict:
        # Classify with test data
        learner.predict(features_path=f"{data_config.output}all_features_test.data", model=f"{data_config.output}model_best_all_features.crfsuite", out_path=f"{data_config.output}result.out")

    parser.evaluate(path=data_config.test)