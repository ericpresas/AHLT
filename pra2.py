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

    train = False

    if train:
        # Get Train and test features
        parser.path_features(output_file=f"{data_config.output}features_train.data", test=False)
        parser.path_features(output_file=f"{data_config.output}features_test.data", test=True)

        # Train the model
        learner.learn(features_path=f"{data_config.output}features_train.data")

    predict = False

    if predict:
        # Classify with test data
        learner.predict(features_path=f"{data_config.output}features_test.data", out_path=f"{data_config.output}result.out")

    parser.evaluate(path=data_config.test)