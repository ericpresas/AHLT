from src.pra6 import Learner
import platform
from config import config_file


resources_section = f"Resources-{platform.system()}"
resources_config = config_file.get(resources_section)
data_config = config_file.get(f"PRA6-Data-{platform.system()}")

if __name__ == "__main__":

    model_file = f"{data_config.output}model2_100.h5"

    learner = Learner(name='cnn', max_len=100, vocab=[], load_model=True)
    learner.load_model(model_file)
    learner.plot_model(path=f"{data_config.output}model2_100.png")
