from sklearn.linear_model import SGDClassifier


class Model(object):
    def __init__(self):
        self.classifier = SGDClassifier()

    def train_batch(self, features, labels, classes_):
        self.classifier.partial_fit(features, labels, classes=classes_)

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def train(self, data_loader_train):
        for i in range(data_loader_train.num_batches):
            print(f"Epoch: {i+1}/{data_loader_train.num_batches}")
            batch_features, batch_labels, _ = data_loader_train.load_features(i)
            self.train_batch(batch_features, batch_labels, data_loader_train.classes_num)

    def predict(self, features):
        pred_labels = self.classifier.predict(features)
        return pred_labels