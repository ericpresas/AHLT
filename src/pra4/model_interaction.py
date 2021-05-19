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


class ModelBinary(object):
    def __init__(self):

        self.classifier = LogisticRegression(random_state=0, verbose=1, solver='sag', max_iter=500, tol=5e-3)

        self.lda = PCA(n_components=400)

    def train_batch(self, features, labels, classes_):
        self.classifier.fit(features, labels)

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def train(self, data_loader_train):
        features = []
        labels = []
        information = []
        for i in range(data_loader_train.num_batches):
            batch_features, batch_labels, info = data_loader_train.load_features(i, is_train=False, binary_class=True)
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

    def predict(self, data_loader):
        predictions = []
        labels = []
        for i in range(data_loader.num_batches):
            print(f"Predicted Batch: {i+1}/{data_loader.num_batches}")
            batch_features, batch_labels, info = data_loader.load_features(i, is_train=False, binary_class=True)
            labels += batch_labels
            #binary_labels = data_loader.load_binary_class(i, binary_path)

            if len(batch_features) > 0:
                batch_features = self.lda.transform(batch_features)
                pred_labels = list(self.predict_batch(batch_features))
                probs = self.classifier.predict_proba(batch_features)
                score_test = f1_score(pred_labels, batch_labels, average='macro')
                print(f"Predicted Batch: {i + 1}/{data_loader.num_batches}, f1_score:{score_test}")
                predictions += pred_labels

        score_test = f1_score(predictions, labels, average='macro')
        print(f"All Predicted f1_score:{score_test}")
        return predictions



    def evaluate(self, files_path, results_path):
        evaluator.evaluate("DDI", files_path, results_path)