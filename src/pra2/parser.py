import os
from xml.dom.minidom import parse, parseString
from ..tokenizer import tokenizer
from eval import evaluator
from .feature_extractor import FeatureExtractor


class Parser(object):
    def __init__(self, path, out_path):
        self.path = path
        self.tokenizer = tokenizer()
        self.out_path = out_path
        self.feature_extractor = FeatureExtractor()

    def path_features(self, output_file):
        with open(output_file, 'w') as outfile:
            # Process each file in directory
            for f in os.listdir(self.path):
                tree = parse(f"{self.path}/{f}")
                sentences = tree.getElementsByTagName("sentence")
                for s in sentences:
                    sid = s.attributes["id"].value
                    stext = s.attributes["text"].value

                    gold = []
                    entities = s.getElementsByTagName("entity")
                    for e in entities:
                        offset = e.attributes["charOffset"].value
                        (start, end) = offset.split(";")[0].split("-")
                        gold.append((int(start), int(end), e.attributes["type"].value))

                    tokens = self.tokenizer.tokenize(stext, ngrams=1)
                    features = self.feature_extractor.extract_features(tokens)

                    for i in range(0, len(tokens)):
                        tag = self.get_tag(tokens[i], gold)
                        print(sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t', file=outfile)

                    print(file=outfile)
