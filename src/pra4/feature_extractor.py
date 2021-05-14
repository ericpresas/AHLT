import os
from xml.dom.minidom import parse
from ..tokenizer import tokenizer
from eval import evaluator
from nltk.stem import PorterStemmer


class FeatureExtractor(object):
    def __init__(self, path, out_path, vocab_path=None):
        self.path = path
        self.tokenizer = tokenizer()
        self.out_path = out_path

        # Tools to extract features
        self.porter_stemmer = PorterStemmer()

        self.vocab_path = vocab_path

    def get_features(self, vocab=False):
        vocabulary = []
        with open(self.out_path, 'w') as outfile:
            for i, f in enumerate(os.listdir(self.path)):
                # print(f"File {i+1}/{len(os.listdir(self.path))}")
                tree = parse(f"{self.path}/{f}")
                sentences = tree.getElementsByTagName("sentence")
                for s in sentences:
                    sid = s.attributes["id"].value
                    stext = s.attributes["text"].value

                    entities = {}
                    ents = s.getElementsByTagName("entity")
                    for e in ents:
                        eid = e.attributes['id'].value
                        entities[eid] = e.attributes['charOffset'].value.split('-')

                    if len(entities) > 1:
                        try:
                            analysis = self.tokenizer.analyze(stext)

                            pairs = s.getElementsByTagName('pair')
                            for p in pairs:
                                id_e1 = p.attributes['e1'].value
                                id_e2 = p.attributes['e2'].value
                                ddi = self.str2bool(p.attributes['ddi'].value)
                                ddi_type = p.attributes['type'].value if ddi else "null"
                                feats = self.extract_features(analysis, entities, id_e1, id_e2)
                                if vocab:
                                    vocabulary += feats
                                    vocabulary = list(set(vocabulary))
                                string_features = "\t".join(feats)
                                if ddi_type != None:
                                    print(sid + "\t" + id_e1 + "\t" + id_e2 + "\t" + ddi_type + "\t" + string_features, file=outfile)
                        except Exception as e:
                            print(f"Error: {stext}")

        if vocab:
            with open(self.vocab_path, 'w') as outfile:
                for feature in vocabulary:
                    print(feature, file=outfile)

    def extract_features(self, analysis, entities, id_e1, id_e2):
        analysis_list = [token for key, token in analysis.items()]
        # WORD FEATURES-------------------------------------------------------------------------------------------------
        # Extract drug1
        drug1_position = [int(position) for position in entities[id_e1]]
        drug1_tokens = list(
            filter(lambda x: (x['start'] <= drug1_position[0]) and (x['end'] >= drug1_position[1]), analysis_list[1:]))

        drug1 = " ".join([token['word'] for token in drug1_tokens])

        # Extract drug2
        drug2_position = [int(position) for position in entities[id_e2]]
        drug2_tokens = list(
            filter(lambda x: (x['start'] <= drug2_position[0]) and (x['end'] >= drug2_position[1]), analysis_list[1:]))

        drug2 = " ".join([token['word'] for token in drug2_tokens])

        # Extract words between drug 1 and drug 2. (Maximum of 3)
        tokens_between = list(filter(lambda x: x['start'] > drug1_position[1] and x['end'] < drug2_position[0], analysis_list[1:]))
        feature_tokens_between = []
        cnt = 0
        for token in tokens_between:
            if token['word'] != ',' and cnt <= 2:
                cnt += 1
                feature_tokens_between += [
                    f"btw{cnt}-word={token['word']}",
                    f"btw{cnt}-lemma={token['lemma']}",
                    f"btw{cnt}-stem={self.porter_stemmer.stem(token['word'])}",
                ]

        # Extract 3 words before drug 1
        tokens_before = list(filter(lambda x: x['start'] < drug1_position[0] and x['word'] != ',', analysis_list[1:]))
        tokens_before = tokens_before[-3:]
        feature_tokens_before = []
        for i, token in enumerate(tokens_before):
            feature_tokens_before += [
                f"bfr{i+1}-word={token['word']}",
                f"bfr{i + 1}-lemma={token['lemma']}",
                f"bfr{i + 1}-stem={self.porter_stemmer.stem(token['word'])}"
            ]

        # Extract 3 words before drug 2
        tokens_after = list(
            filter(lambda x: x['end'] > drug2_position[1] and x['word'] != ',', analysis_list[1:]))
        tokens_after = tokens_before[:3]
        feature_tokens_after = []
        for i, token in enumerate(tokens_after):
            feature_tokens_after += [
                f"aft{i + 1}-word={token['word']}",
                f"aft{i + 1}-lemma={token['lemma']}",
                f"aft{i + 1}-stem={self.porter_stemmer.stem(token['word'])}"
            ]

        features = [
            f"drug1={drug1}",
            f"drug2={drug2}"
        ]

        features += feature_tokens_between
        features += feature_tokens_before
        features += feature_tokens_after

        return features

    def open_features(self):
        with open(self.out_path, encoding="utf8") as f:
            lines = f.readlines()
            all_features = []
            for i, line in enumerate(lines):
                features = line.strip().split('\t')
                info = features[:4]
                features = features[4:]
                all_features += features

            unique_features = list(set(all_features))

            feature_vector = []
            labels = []
            for i, line in enumerate(lines):
                features = line.strip().split('\t')
                info = features[:4]
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

        return feature_vector, labels


    @staticmethod
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    def evaluate(self, path):
        evaluator.evaluate("DDI", path, self.out_path)








