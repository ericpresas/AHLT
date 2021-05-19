from sklearn import preprocessing
from collections import Counter


class DataLoader(object):
    def __init__(self, path, batch_size, vocabuary_path, binary_path=None, label_encoder=None, make_vocab=False, vocab_size=1000):
        self.path = path
        self.batch_size = batch_size
        with open(self.path, encoding="utf8") as f:
            lines = f.readlines()
            if make_vocab:
                vocabulary = []
                for line in lines:
                    line = line.strip().split('\t')
                    features = line[4:]
                    vocabulary += features

            self.num_batches = int(len(lines) / self.batch_size)
            all_classes = [line.split('\t')[3] for line in lines]
            all_classes = list(filter(lambda x: x != 'null', all_classes))
            all_classes = sorted(list(set(all_classes)))

        with open('out/pra4/test_features.out', encoding="utf-8") as f:
            lines = f.readlines()
            if make_vocab:
                for line in lines:
                    line = line.strip().split('\t')
                    features = line[4:]
                    vocabulary += features

                vocabulary = Counter(vocabulary).most_common(vocab_size)
                with open(vocabuary_path, 'w') as outfile:
                    for feature in vocabulary:
                        print(feature[0], file=outfile)

        self.vocab = self.load_vocabulary(vocabuary_path)
        self.classes_ = all_classes
        #self.classes_ = all_classes
        self.classes_num = [i for i, class_ in enumerate(all_classes)]
        self.binary_path = binary_path
        if label_encoder is not None:
            self.label_encoder = label_encoder
        else:
            self.label_encoder = preprocessing.LabelEncoder()

    @staticmethod
    def load_vocabulary(vocabulary_path):
        with open(vocabulary_path, encoding="utf8") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        return lines

    def train_encoder(self):
        with open(self.path, encoding="utf8") as f:
            lines = f.readlines()
            labels = []
            for i, line in enumerate(lines):
                features = line.strip().split('\t')
                labels.append(features[3])

            self.label_encoder.fit(labels)

    def load_features(self, iteration, is_train=False, binary_class=False):
        with open(self.path, encoding="utf8") as f:
            lines = f.readlines()
            all_features = []
            start = self.batch_size*iteration
            lines = lines[start:start + self.batch_size]

            unique_features = self.vocab

            feature_vector = []
            labels = []
            info_vector = []
            for i, line in enumerate(lines):
                features = line.strip().split('\t')
                info = features[:4]
                if is_train and info[-1]!='null':
                    info_vector.append(info)
                    labels.append(info[-1])
                    features = features[4:]
                    feature_line = []
                    for index, feature in enumerate(unique_features):
                        if feature in features:
                            feature_line.append(1)
                        else:
                            feature_line.append(0)
                    feature_vector.append(feature_line)
                elif not is_train:
                    info_vector.append(info)
                    labels.append(info[-1])
                    features = features[4:]
                    feature_line = []
                    for index, feature in enumerate(unique_features):
                        if feature in features:
                            feature_line.append(1)
                        else:
                            feature_line.append(0)
                    feature_vector.append(feature_line)

            if binary_class:
                for i, label in enumerate(labels):
                    if label != 'null':
                        labels[i] = 1
                    else:
                        labels[i] = 0
            else:
                labels = self.label_encoder.transform(labels)
        return feature_vector, labels, info_vector

    def load_binary_class(self, iteration, path):
        with open(self.binary_path, encoding="utf8") as f:
            lines = f.readlines()
            start = self.batch_size * iteration
            lines = lines[start:start + self.batch_size]

            binary_labels = []

            for i, line in enumerate(lines):
                info = line.strip().split('|')
                binary_labels.append(info[-1])

        return binary_labels


