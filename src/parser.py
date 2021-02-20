import os
from xml.dom.minidom import parse, parseString
from .tokenizer import tokenizer


class Parser(object):
    def __init__(self, path, out_path):
        self.path = path
        self.tokenizer = tokenizer
        self.out_path = out_path

    def extract_entities(self, tokens):
        entities = []
        return entities

    def parse_dir(self):
        for f in os.listdir(self.path):
            tree = parse(f"{self.path}/{f}")
            sentences = tree.getElementsByTagName("sentence")
            for s in sentences:
                sid = s.attributes["id"].value
                stext = s.attributes["text"].value
                tokens = self.tokenizer.tokenize(stext)
                entities = self.extract_entities(tokens)

                for e in entities:
                    print(sid + "|" + e["offset"] + "|" + e["text"] + "|" + e["type"], file=self.out_path)

        #evaluator.evaluate("NER", datadir, outfile)

