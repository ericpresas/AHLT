from ..tokenizer import tokenizer
from xml.dom.minidom import parse
import os
import numpy as np

class DataLoader(object):
    def __init__(self):
        self.tokenizer = tokenizer()

    def load_data(self, path):
        #with open(self.out_path, 'w') as outfile:
        tokenized_dataset = []
        my_dict = {}
        for i, f in enumerate(os.listdir(path)):
            print(f"File {i+1}/{len(os.listdir(path))}")
            tree = parse(path + "/" + f)
            # process each sentence in the file
            sentences = tree.getElementsByTagName("sentence")
            for s in sentences:
                sid = s.attributes["id"].value  # get sentence id
                stext = s.attributes["text"].value  # get sentence text

                tokens = self.tokenizer.tokenize(stext, ngrams=1)
                if len(tokens) > 1:

                    # load sentence entities into a dictionary
                    entities = s.getElementsByTagName("entity")
                    gold = []
                    values = []
                    for e in entities:
                        # for discontinuous entities , we only get the first span
                        offset = e.attributes["charOffset"].value
                        (start, end) = offset.split(";")[0].split("-")
                        gold.append((int(start), int(end), e.attributes["type"].value))

                        print((sid, int(start), int(end), e.attributes["type"].value))

                    for i in range(0, len(tokens)):
                        tag = self.get_tag(tokens[i], gold)
                        values.append((tokens[i][0], tokens[i][1], tokens[i][2], tag))

                    my_dict[sid] = values

        return my_dict

    def get_tag(self, token, gold):
        text, start, end = token
        pos = None
        type_pos = None
        for entity in gold:
            entity_start, entity_end, type = entity
            if start >= entity_start and start <= entity_end:
                pos = 'B' if entity_start == start else 'I'
                type_pos = type
        if pos is None:
            pos = 'O'

        if type_pos is None:
            result = 'O'
        else:
            result = f"{pos}-{type_pos}"
        return result


    def create_indexs(self, dataset, max_length):
        labels = {
            "<PAD>": 0,
            "B-group": 1,
            "B-drug_n": 2,
            "B-drug": 3,
            "B-brand": 4,
            "I-group": 5,
            "I-drug_n": 6,
            "I-drug": 7,
            "I-brand": 8,
            "O": 9
        }

        words = {
            '<PAD>': 0,
            '<UNK>': 1
        }

        pref = {
            '<PAD>': 0,
            '<UNK>': 1
        }

        suf = {
            '<PAD>': 0,
            '<UNK>': 1
        }

        count_words = 2
        count_suf = 2
        count_pref = 2
        for key, sentence in dataset.items():
            if len(sentence) > max_length:
                sentence = sentence[:max_length]

            for token in sentence:
                pref_chars = token[0][:4]
                suf_chars = token[0][-4:]

                if token[0] not in words:
                    words[token[0]] = count_words
                    count_words += 1

                if pref_chars not in pref:
                    pref[pref_chars] = count_pref
                    count_pref += 1

                if suf_chars not in suf:
                    suf[suf_chars] = count_suf
                    count_suf += 1

        return {
            'words': words,
            'suf': suf,
            'pref': pref,
            'labels': labels,
            'maxlen': max_length
        }

    def encode_words(self, dataset, indexs):
        embeddings_words = []
        embeddings_suf = []
        embeddings_pref = []
        for key, sentence in dataset.items():
            if len(sentence) > indexs['maxlen']:
                sentence = sentence[:indexs['maxlen']]
            words = []
            pref = []
            suf = []
            for token in sentence:
                pref_chars = token[0][:4]
                suf_chars = token[0][-4:]

                if token[0] in indexs['words']:
                    words.append(indexs['words'][token[0]])
                else:
                    words.append(indexs['words']['<UNK>'])

                if pref_chars in indexs['pref']:
                    pref.append(indexs['pref'][pref_chars])
                else:
                    pref.append(indexs['pref']['<UNK>'])

                if suf_chars in indexs['suf']:
                    suf.append(indexs['suf'][suf_chars])
                else:
                    suf.append(indexs['suf']['<UNK>'])

            if len(sentence) < indexs['maxlen']:
                padding_words = [indexs['words']['<PAD>'] for x in range(indexs['maxlen'] - len(sentence))]
                padding_pref = [indexs['pref']['<PAD>'] for x in range(indexs['maxlen'] - len(sentence))]
                padding_suf = [indexs['suf']['<PAD>'] for x in range(indexs['maxlen'] - len(sentence))]
                words += padding_words
                pref += padding_pref
                suf += padding_suf

            embeddings_words.append(words)
            embeddings_pref.append(pref)
            embeddings_suf.append(suf)


        return {
            'words': embeddings_words,
            'suf': embeddings_suf,
            'pref': embeddings_pref
        }

    def encode_labels(self, dataset, indexs):
        labels = []
        for key, sentence in dataset.items():
            if len(sentence) > indexs['maxlen']:
                sentence = sentence[:indexs['maxlen']]
            labels_tmp = []
            for token in sentence:
                label = token[3]
                labels_tmp.append(indexs['labels'][label])

            if len(sentence) < indexs['maxlen']:
                padding_labels = [indexs['labels']['<PAD>'] for x in range(indexs['maxlen'] - len(sentence))]
                labels_tmp += padding_labels

            labels.append(labels_tmp)
        return labels