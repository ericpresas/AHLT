import os
from xml.dom.minidom import parse, parseString
from ..tokenizer import tokenizer
from eval import evaluator
from .feature_extractor import FeatureExtractor
from .learner import Learner
import pycrfsuite
from .utils import Utils


class Parser(object):
    def __init__(self, data_paths, resources_paths, out_path):
        self.path_train = data_paths.train
        self.path_test = data_paths.test
        self.tokenizer = tokenizer()
        self.out_path = out_path
        self.feature_extractor = FeatureExtractor(utils=Utils(drugbank_path=resources_paths.drugbank, hsdb_path=resources_paths.hsdb))

    def path_features(self, output_file, test=True):
        path = self.path_train
        if test:
            path = self.path_test
        with open(f"{output_file}", 'w') as outfile:
            # Process each file in directory
            for count, f in enumerate(os.listdir(path)):
                print(f'processed {count}/{len(os.listdir(path))}')
                tree = parse(f"{path}/{f}")
                sentences = tree.getElementsByTagName("sentence")

                for s in sentences:
                    sid = s.attributes["id"].value
                    stext = s.attributes["text"].value
                    gold = []
                    tags = []
                    entities = s.getElementsByTagName("entity")
                    for e in entities:
                        offset = e.attributes["charOffset"].value
                        (start, end) = offset.split(";")[0].split("-")
                        tags.append(e.attributes["type"].value)
                        gold.append((int(start), int(end), e.attributes["type"].value))

                    tokens = self.tokenizer.tokenize(stext, ngrams=1)
                    features = self.feature_extractor.extract_features(tokens, stext)
                    for i in range(0, len(tokens)):
                        tag = self.feature_extractor.get_tag(tokens[i], gold)
                        print(sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t',
                                  file=outfile)

                    print(file=outfile)

    def evaluate(self, path):
        evaluator.evaluate("NER", path, self.out_path)








