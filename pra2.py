from src.pra2 import Parser

if __name__ == "__main__":
    path = "data/devel/"
    out_path = "out/pra2/result.out"

    parser = Parser(path=path, out_path=out_path)
    parser.path_features('out/pra2/features.data')