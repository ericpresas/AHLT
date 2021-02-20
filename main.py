from src import Parser

if __name__ == "__main__":
    path = "data/devel/"
    out_path = "out/"

    parser = Parser(path=path, out_path=out_path)
    parser.parse_dir()
