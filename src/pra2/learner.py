from itertools import chain
import nltk
import sklearn
import pycrfsuite


class Learner(object):
    def __init__(self, out_path):
        self.out_path = out_path

    @staticmethod
    def read_features(features_path):
        all_tags = []
        features = []
        all_ids = []
        with open(features_path, encoding="utf8") as f:
            lines = f.readlines()
            token_features = []
            tags = []
            ids = []
            for i, line in enumerate(lines):
                text = line.strip().split()

                if len(text) > 0:

                    tags.append(text[4])
                    form, suf4, next, prev = text[-4:]
                    token_features.append([form, suf4, next, prev])
                    ids.append((text[0], text[1], text[2], text[3]))
                    print(text)
                else:
                    features.append(token_features)
                    token_features = []
                    all_tags.append(tags)
                    tags = []
                    all_ids.append(ids)
                    ids = []

        return all_tags, features, all_ids

    def learn(self, features_path):
        tags, features, _ = self.read_features(features_path=features_path)
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

    def predict(self, features_path, out_path):
        tags, features, ids = self.read_features(features_path=features_path)
        tagger = pycrfsuite.Tagger()
        tagger.open(self.out_path)
        classified_data = []
        with open(out_path, 'w') as outfile:
            for feat_list, tag_list, id_list in zip(features, tags, ids):
                preds = tagger.tag(feat_list)
                start_index = 0
                end_index = 0
                tmp_class = ''
                text_class = ''
                for j, pred_label in enumerate(preds):
                    if 'B' in pred_label:
                        tmp_class = pred_label.split('-')[1]
                        text_class += feat_list[j][0].split('=')[1]
                        start_index = id_list[j][2]
                        end_index = id_list[j][3]
                    elif 'I' in pred_label:
                        if pred_label.split('-')[1] == tmp_class:
                            end_index = id_list[j][3]
                            text_class += ' ' + feat_list[j][0].split('=')[1]
                    else:
                        if tmp_class != '':
                            classified_data.append((id_list[j][0], start_index, end_index, text_class, tmp_class))
                            print(f"{id_list[j][0]}|{start_index}-{end_index}|{text_class}|{tmp_class}")
                            print(f"{id_list[j][0]}|{start_index}-{end_index}|{text_class}|{tmp_class}", file=outfile)
                        start_index = 0
                        end_index = 0
                        tmp_class = ''
                        text_class = ''