from src.pra2 import Parser
from src.pra2 import Learner
import platform
from config import config_file

from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite

resources_section = f"Resources-{platform.system()}"
resources_config = config_file.get(resources_section)
data_config = config_file.get(f"PRA2-Data-{platform.system()}")

if __name__ == "__main__":
    out_path = f"{data_config.output}result.out"

    parser = Parser(path=data_config.devel, out_path=out_path)
    parser.path_features(f"{data_config.output}features.data")
    learner = Learner(f"{data_config.output}training_data.crfsuite")
    learner.learn(f"{data_config.output}features.data")