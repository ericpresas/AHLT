import os
from xml.dom.minidom import parse
from ..tokenizer import tokenizer
from eval import evaluator
from nltk.stem import PorterStemmer
from .utils import Utils
from collections import Counter
import networkx as nx
from nltk.sentiment.util import mark_negation


class FeatureExtractor(object):
    def __init__(self, path, out_path, vocab_path=None):
        self.path = path
        self.tokenizer = tokenizer()
        self.out_path = out_path

        # Tools to extract features
        self.porter_stemmer = PorterStemmer()

        self.vocab_path = vocab_path

        self.effect_verbs = set(
            ["administer", "potentiate", "prevent", "stimulate"" stimulated", "antagonize", "antagonized", "reduce",
             "reduced", "increase", "increased", "decreased", "enhance", "enhanced"])
        self.mechanism_verbs = set(
            ["reduce", "reduced", "increase", "increased", "decreased", "decrease", "enhance", "enhanced"])
        self.int_verbs = set(["interact", "interaction", "interactions", "interfere"])
        self.advise_verbs = set(["advise", "advised", "caution", "Caution", "consideration", "considerations", "should be"])

    def get_features(self, vocab=False, is_train=False):
        vocabulary = []
        print("Get features")
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
                    char_offset = []
                    for e in ents:
                        eid = e.attributes['id'].value
                        e.attributes['charOffset'].value = e.attributes['charOffset'].value.split(';')[0]
                        entities[eid] = (e.attributes['charOffset'].value.split('-'), e.attributes['type'].value)
                        char_offset.append(e.attributes['charOffset'])
                    if len(entities) > 1:
                        stext = stext.replace('%', 'p')
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
                            string_features = "\t".join(feats)
                            print(sid + "\t" + id_e1 + "\t" + id_e2 + "\t" + ddi_type + "\t" + string_features, file=outfile)


    def extract_features(self, tree, entities, id_e1, id_e2):
        analysis = tree['raw']
        triples = tree['triples']
        nx_graph = tree['graph']


        analysis_list = [token for key, token in analysis.items()]


        # WORD FEATURES-------------------------------------------------------------------------------------------------
        # Extract drug1
        #start_drug1 = entities[id_e1].split(";")
        drug1_position = [int(position) for position in entities[id_e1][0]]
        drug1_type = entities[id_e1][1]
        drug1_tokens = list(
            filter(lambda x: (x['start'] >= drug1_position[0]) and (x['end'] <= drug1_position[1]), analysis_list[1:]))

        drug1 = " ".join([token['word'] for token in drug1_tokens])

        # Extract drug2
        drug2_type = entities[id_e2][1]
        drug2_position = [int(position) for position in entities[id_e2][0]]
        drug2_tokens = list(
            filter(lambda x: (x['start'] >= drug2_position[0]) and (x['end'] <= drug2_position[1]), analysis_list[1:]))

        drug2 = " ".join([token['word'] for token in drug2_tokens])

        # Other entities



        features = [
            f"drug1={drug1}",
            f"drug2={drug2}",
            f"entities-type={drug1_type}-{drug2_type}"
        ]


        distance_between = Utils.distance_between(drug1_position, drug2_position)

        if drug1 != drug2:
            btw_position = Utils.start_end_between(drug1_position, drug2_position)
            tokens_between = list(
                filter(lambda x: x['start'] > btw_position[0] and x['end'] < btw_position[1], analysis_list[1:]))

            sentence = [token['word'] for token in analysis_list if token['word'] is not None]
            if bool(self.effect_verbs.intersection(sentence)):
                features.append(f"effect-verb-between=true")

            if bool(self.mechanism_verbs.intersection(sentence)):
                features.append(f"mechanism-verb-between=true")

            if bool(self.int_verbs.intersection(sentence)):
                features.append(f"int-verb-between=true")

            if bool(self.advise_verbs.intersection(sentence)):
                features.append(f"advise-verb-between=true")

            # Extract words between drug 1 and drug 2. (Maximum of 3)
            btw_position = Utils.start_end_between(drug1_position, drug2_position)
            tokens_between = list(filter(lambda x: x['start'] > btw_position[0] and x['end'] < btw_position[1], analysis_list[1:]))
            feature_tokens_between = []
            cnt = 0
            for token in tokens_between:
                if not Utils.is_special_char(token['word']) and not Utils.is_numeric(token['word']):
                    cnt += 1
                    feature_tokens_between += [
                        f"btw-word={token['word']}",
                        f"btw-lemma={token['lemma']}",
                        f"btw-stem={self.porter_stemmer.stem(token['word'])}",
                        f"btw-tag={token['tag']}"
                    ]

            # Extract 3 words before drug 1
            before_position = Utils.before_position(drug1_position, drug2_position)
            tokens_before = list(filter(lambda x: x['start'] < before_position and x['word'] != ',', analysis_list[1:]))
            tokens_before = tokens_before[-3:]
            feature_tokens_before = []
            cnt = 0
            for i, token in enumerate(tokens_before):
                if not Utils.is_special_char(token['word']) and not Utils.is_numeric(token['word']):
                    cnt += 1
                    feature_tokens_before += [
                        f"bfr-word={token['word']}",
                        f"bfr-lemma={token['lemma']}",
                        f"bfr-stem={self.porter_stemmer.stem(token['word'])}",
                        f"bfr-tag={token['tag']}"
                    ]

            # Extract 3 words before drug 2
            after_position = Utils.after_position(drug1_position, drug2_position)
            tokens_after = list(
                filter(lambda x: x['end'] > after_position and x['word'] != ',', analysis_list[1:]))
            tokens_after = tokens_before[:3]
            feature_tokens_after = []
            cnt = 0
            for i, token in enumerate(tokens_after):
                if not Utils.is_special_char(token['word']) and not Utils.is_numeric(token['word']):
                    cnt += 1
                    feature_tokens_after += [
                        f"aft-word={token['word']}",
                        f"aft-lemma={token['lemma']}",
                        f"aft-stem={self.porter_stemmer.stem(token['word'])}",
                        f"aft-tag={token['tag']}"
                    ]

            verb_tokens = list(filter(lambda x: 'VB' in x['tag'], analysis_list))
            if len(verb_tokens) > 0:
                for verb_token in verb_tokens:
                    feature_verb_tag = f"vb-after-tag={verb_token['tag']}"
                    feature_verb_rel = f"vb-after-rel={verb_token['rel']}"
                    feature_verb_lemma = f"vb-after-lemma={verb_token['lemma']}"
                    feature_verb_stem = f"vb-after-stem={self.porter_stemmer.stem(verb_token['word'])}"
                    if verb_token['start'] > btw_position[0] and verb_token['end'] < btw_position[1]:
                        feature_verb_tag = f"vb-btw-tag={verb_token['tag']}"
                        feature_verb_rel = f"vb-btw-rel={verb_token['rel']}"
                        feature_verb_lemma = f"vb-btw-lemma={verb_token['lemma']}"
                        feature_verb_stem = f"vb-btw-stem={self.porter_stemmer.stem(verb_token['word'])}"
                    elif verb_token['end'] < before_position:
                        feature_verb_tag = f"vb-before-tag={verb_token['tag']}"
                        feature_verb_rel = f"vb-before-rel={verb_token['rel']}"
                        feature_verb_lemma = f"vb-before-lemma={verb_token['lemma']}"
                        feature_verb_stem = f"vb-before-stem={self.porter_stemmer.stem(verb_token['word'])}"

                    features += [feature_verb_tag, feature_verb_lemma, feature_verb_rel, feature_verb_stem]

            # Shortest path

            paths = []
            path = []
            verb_word = ''

            if len(drug1_tokens) > 0 and len(drug2_tokens) > 0:
                sht1_path = nx.shortest_path(nx_graph, source=analysis_list.index(drug1_tokens[-1]))
                sht2_path = nx.shortest_path(nx_graph, source=analysis_list.index(drug2_tokens[-1]))

                path_nodes = list(sht1_path.keys() & sht2_path.keys())
                path_string = ''
                if len(path_nodes) > 0:
                    path_nodes = path_nodes[0]
                    path1_list = [analysis_list[node] for node in sht1_path[path_nodes]]
                    path2_list = [analysis_list[node] for node in sht2_path[path_nodes]]
                    if drug1_tokens[-1]['word'] == path1_list[-1]['word']:
                        # Direct relation
                        if len(path1_list) > 1:
                            for pos in range(len(path1_list) - 1):
                                path_string += f"{path1_list[pos]['rel']}>"
                        elif len(path2_list) > 1:
                            for pos in range(len(path1_list) - 1):
                                path_string += f"{path1_list[pos]['rel']}>"
                    else:
                        for pos in range(len(path1_list) - 1):
                            path_string += f"{path1_list[pos]['rel']}>"

                        path_string += f"{path1_list[-1]['lemma']}>"

                        for pos in range(len(path2_list) - 1):
                            path_string += f"{path2_list[pos]['rel']}>"


                    print(path_string)
                    features += [f"path={path_string}"]

            features += feature_tokens_between
            features += feature_tokens_before
            features += feature_tokens_after

        word_list = [token['word'] for key, token in analysis.items() if token['word'] is not None]
        negation_word_list = mark_negation(word_list)
        after_negation_word_list = list(filter(lambda x: '_NEG' in x, negation_word_list))
        if len(after_negation_word_list) > 0:
            first_index = negation_word_list.index(after_negation_word_list[0])
            negation_word = negation_word_list[first_index-1]
            features.append(f"neg={negation_word}")
            for negation_word in after_negation_word_list:
                word = negation_word.split('_')[0]
                if not Utils.is_special_char(word):
                    features.append(f"aft-neg={word}")
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








