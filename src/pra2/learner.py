from itertools import chain
import nltk
import sklearn
import pycrfsuite


class Learner(object):
    def __init__(self, out_path):
        self.out_path = out_path
        pass


    def learn(self, features_path):
        all_tags = []
        features = []
        with open(features_path, encoding="utf8") as f:
            lines = f.readlines()
            token_features = []
            tags = []
            for i, line in enumerate(lines):
                text = line.strip().split()

                if len(text) > 0:

                    tags.append(text[4])
                    form, suf4, next, prev = text[-4:]
                    token_features.append([form, suf4, next, prev])

                    print(text)
                else:
                    features.append(token_features)
                    token_features = []
                    all_tags.append(tags)
                    tags = []

        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(features, tags):
            trainer.append(xseq, yseq)
        trainer.set_params({
            'c1': 1.0,  # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        trainer.train(self.out_path)
