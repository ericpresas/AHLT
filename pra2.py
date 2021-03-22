from src.pra2 import Parser
from src.pra2 import Learner
import platform
from config import config_file

resources_section = f"Resources-{platform.system()}"
resources_config = config_file.get(resources_section)
data_config = config_file.get(f"PRA2-Data-{platform.system()}")

if __name__ == "__main__":
    out_path = f"{data_config.output}result.out"

    parser = Parser(data_paths=data_config, out_path=out_path)
    learner = Learner(out_path=f"{data_config.output}training_data.crfsuite")

    extract_features = False

    if extract_features:
        # Get Train and test features
        parser.path_features(output_file=f"{data_config.output}features_train.data", test=False)
        parser.path_features(output_file=f"{data_config.output}features_test.data", test=True)

    estimate_params = False

    # Para usar el estimador downgrade scikit : pip install -U 'scikit-learn<0.24'
    if estimate_params:

        labels = ['B-drug', 'I-drug', 'B-drug_n', 'I-drug_n', 'B-group', 'I-group', 'B-brand', 'I-brand']
        best_params = learner.estimate_best_params(features_path=f"{data_config.output}features_train.data", labels=labels)

    else:

        best_params = {
            "c1": 0.025,
            "c2": 0.06,
            "linesearch": 'MoreThuente'
        }

    # Train the model
    learner.learn(features_path=f"{data_config.output}features_train.data", params=best_params)

    predict = True

    if predict:
        # Classify with test data
        learner.predict(features_path=f"{data_config.output}features_test.data", out_path=f"{data_config.output}result.out")

    parser.evaluate(path=data_config.test)