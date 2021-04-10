from src.pra3 import Parser
import platform
from config import config_file

resources_section = f"Resources-{platform.system()}"
resources_config = config_file.get(resources_section)
data_config = config_file.get(f"PRA3-Data-{platform.system()}")

if __name__ == "__main__":
    out_path = f"{data_config.output}result.out"

    parser = Parser(path=data_config.train, out_path=out_path)
    parser.path_process()
    parser.evaluate(path=data_config.test)
    print('a')
