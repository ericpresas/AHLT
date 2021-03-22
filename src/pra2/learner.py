import sklearn
import pycrfsuite
from sklearn.model_selection import GridSearchCV
import sklearn_crfsuite
from sklearn.metrics import classification_report
from sklearn_crfsuite import metrics
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats


class Learner(object):
    def __init__(self, out_path):
        self.out_path = out_path

    @staticmethod
    def features2dict(features):
        return {feature.split('=')[0]: feature.split('=')[1] for feature in features}

    def read_features(self, features_path, dict_=False):
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

                    form, form_lower, form_isupper, form_istitle, form_isdigit, suf4, next, next_lower, prev, prev_lower = text[-10:]
                    ids.append((text[0], text[1], text[2], text[3]))

                    if dict_:
                        token_features.append(self.features2dict(text[-10:]))
                    else:
                        token_features.append([form, form_lower, form_isupper, form_istitle, form_isdigit, suf4, next, next_lower, prev, prev_lower])
                    #print(text)
                else:
                    features.append(token_features)
                    token_features = []
                    all_tags.append(tags)
                    tags = []
                    all_ids.append(ids)
                    ids = []

        return all_tags, features, all_ids

    def learn(self, features_path, params={"c1": 1.0, "c2": 1e-3}):
        tags, features, _ = self.read_features(features_path=features_path)
        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(features, tags):
            trainer.append(xseq, yseq)
        trainer.set_params({
            'c1': params['c1'],  # coefficient for L1 penalty
            'c2': params['c2'],  # coefficient for L2 penalty
            'max_iterations': 100,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        trainer.train(self.out_path)

    def estimate_best_params(self, features_path, labels):
        #https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#hyperparameter-optimization
        tags, features, _ = self.read_features(features_path=features_path, dict_=True)
        crf = sklearn_crfsuite.CRF(
            max_iterations=100,
            all_possible_transitions=True
        )
        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
            'algorithm': ['lbfgs']
        }

        # use the same metric for evaluation
        f1_scorer = make_scorer(metrics.flat_f1_score,
                                average='weighted', labels=labels)

        # search
        rs = RandomizedSearchCV(crf, params_space,
                                cv=3,
                                verbose=1,
                                n_jobs=-1,
                                n_iter=50,
                                scoring=f1_scorer)
        rs.fit(features, tags)

        # crf = rs.best_estimator_
        print('best params:', rs.best_params_)
        print('best CV score:', rs.best_score_)
        print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

        return rs.best_params_

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
                            #print(f"{id_list[j][0]}|{start_index}-{end_index}|{text_class}|{tmp_class}")
                            print(f"{id_list[j][0]}|{start_index}-{end_index}|{text_class}|{tmp_class}", file=outfile)
                        start_index = 0
                        end_index = 0
                        tmp_class = ''
                        text_class = ''