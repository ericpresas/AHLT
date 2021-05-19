from sklearn.linear_model import SGDClassifier
from eval import evaluator
from sklearn.metrics import f1_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA


class Model(object):
    def __init__(self):
        self.classifier = SGDClassifier(loss="modified_huber",
                                penalty="l2",
                                alpha=0.001,
                                max_iter=1000,
                                tol=None,
                                shuffle=True,
                                verbose=1,
                                learning_rate='optimal',
                                eta0=0.1,
                                n_iter_no_change=3,
                                early_stopping=True)

        """self.classifier = SGDClassifier(alpha=0.001, average=True, class_weight=None,
              early_stopping=False, epsilon=0.1, eta0=0.01, fit_intercept=True,
              l1_ratio=0.2, learning_rate='adaptive', loss='log', max_iter=10,
              n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.05,
              random_state=None, shuffle=True, tol=0.001,
              validation_fraction=0.2, verbose=1, warm_start=True)"""

        """self.classifier = SVC(gamma='auto', kernel='linear')"""
        """self.classifier = LogisticRegression(random_state=0, verbose=1, solver='sag', max_iter=500, tol=1e-4)"""

        self.lda = PCA(n_components=100)

    def train_batch(self, features, labels, classes_):
        self.classifier.fit(features, labels)

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def train(self, data_loader_train, data_loader_devel):
        features = []
        labels = []
        information = []
        for i in range(data_loader_train.num_batches):
            batch_features, batch_labels, info = data_loader_train.load_features(i, is_train=True)
            information += info
            features += batch_features
            labels += list(batch_labels)
            print(f"Loader: {i + 1}/{data_loader_train.num_batches}")

        features = self.lda.fit_transform(features, labels)
        self.train_batch(features, labels, data_loader_train.classes_num)
        pred_labels = self.classifier.predict(features)
        with open('out/pra4/results_train.out', 'w') as outfile:
            for i, prediction in enumerate(pred_labels):
                sid = information[i][0]
                id_e1 = information[i][1]
                id_e2 = information[i][2]

                if data_loader_train.label_encoder.classes_[prediction] != 'null':
                    print(sid + "|" + id_e1 + "|" + id_e2 + "|" + str(data_loader_train.label_encoder.classes_[prediction]),
                          file=outfile)
        score_test = f1_score(pred_labels, labels, average='macro')
        print(f"Predicted train f1_score:{score_test}")


    def predict_batch(self, features):
        pred_labels = self.classifier.predict(features)
        return pred_labels

    def predict(self, data_loader, path, is_interaction_labels):
        with open(path, 'w') as outfile:
            for i in range(data_loader.num_batches):
                print(f"Predicted Batch: {i+1}/{data_loader.num_batches}")
                batch_features, batch_labels, info = data_loader.load_features(i, is_train=False)
                start = data_loader.batch_size * i
                is_interaction = is_interaction_labels[start:start + data_loader.batch_size]
                #binary_labels = data_loader.load_binary_class(i, binary_path)

                if len(batch_features) > 0:
                    batch_features = self.lda.transform(batch_features)
                    pred_labels = self.predict_batch(batch_features)
                    probs = self.classifier.predict_proba(batch_features)
                    score_test = f1_score(pred_labels, batch_labels, average='macro')
                    print(f"Predicted Batch: {i + 1}/{data_loader.num_batches}, f1_score:{score_test}")
                    for i, prediction in enumerate(pred_labels):
                        sid = info[i][0]
                        id_e1 = info[i][1]
                        id_e2 = info[i][2]
                        probability = list(probs[i])

                        if max(probability) >= 0.1 and is_interaction_labels[i]==1:
                            if data_loader.label_encoder.classes_[prediction] != 'null':
                                print(sid + "|" + id_e1 + "|" + id_e2 + "|" + str(data_loader.label_encoder.classes_[prediction]), file=outfile)
                        else:
                            pass
                            #print(f"{info[i][3]} -> {data_loader.label_encoder.classes_[prediction]} -> {probs[i]}")



    def evaluate(self, files_path, results_path):
        evaluator.evaluate("DDI", files_path, results_path)