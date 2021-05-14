class DataLoader(object):
    def __init__(self, path, batch_size, vocabuary_path):
        self.path = path
        self.vocab = self.load_vocabulary(vocabuary_path)
        self.batch_size = batch_size
        with open(self.path, encoding="utf8") as f:
            lines = f.readlines()
            all_classes = [line.split('\t')[3] for line in lines]
            all_classes = sorted(list(set(all_classes)))
        self.classes_ = all_classes
        self.classes_num = [i for i, class_ in enumerate(all_classes)]
        self.num_batches = int(len(lines)/self.batch_size)

    @staticmethod
    def load_vocabulary(vocabulary_path):
        with open(vocabulary_path, encoding="utf8") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        return lines

    def load_features(self, iteration):
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

            unique_labels = list(set(labels))
            labels = [unique_labels.index(label) for label in labels]

        return feature_vector, labels, info_vector

