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
        for i, f in enumerate(os.listdir(path)):
            print(f"File {i+1}/{len(os.listdir(path))}")
            tree = parse(f"{path}/{f}")
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
                        tokenized_pair_sentence = self.build_tokens(analysis, entities, id_e1, id_e2)
                        ddi = self.str2bool(p.attributes['ddi'].value)
                        ddi_type = p.attributes['type'].value if ddi else "null"
                        tokenized_dataset.append([sid, id_e1, id_e2, ddi_type, tokenized_pair_sentence])
                        #print(sid + "\t" + id_e1 + "\t" + id_e2 + "\t" + ddi_type + "\t" + string_features, file=outfile)
        return tokenized_dataset

    def create_indexs(self, dataset, max_length):
        labels = {'null': 0, 'mechanism': 1, 'advise': 2, 'effect': 3, 'int': 4}

        words = {
            '<PAD>': 0,
            '<UNK>': 1
        }
        count = 2
        for interaction in dataset:
            tokens = interaction[4]
            if len(tokens) > max_length:
                tokens = self.find_between_tags(tokens, start_tag=('<DRUG1>', '<DRUG1>', '<DRUG1>'), end_tag=('<DRUG2>', '<DRUG2>', '<DRUG2>'))

            if len(tokens) > max_length:
                #print(f"Sentence: {interaction[:4]} - {len(tokens)}")
                tokens = tokens[:100]

            for token in tokens:
                if token not in words:
                    words[token] = count
                    count += 1

        return {
            'words': words,
            'labels': labels,
            'maxlen': max_length
        }

    def encode_words(self, dataset, indexs):
        interactions_embeddings = []
        for interaction in dataset:
            tokens = interaction[4]
            if len(tokens) > indexs['maxlen']:
                tokens = self.find_between_tags(tokens, start_tag=('<DRUG1>', '<DRUG1>', '<DRUG1>'), end_tag=('<DRUG2>', '<DRUG2>', '<DRUG2>'))

            if len(tokens) > indexs['maxlen']:
                tokens = tokens[:100]

            interaction_embedding = []
            for token in tokens:
                if token in indexs['words']:
                    interaction_embedding.append(indexs['words'][token])
                else:
                    interaction_embedding.append(indexs['words']['<UNK>'])

            if len(tokens) < indexs['maxlen']:
                additional_padding = [indexs['words']['<PAD>'] for x in range(indexs['maxlen'] - len(tokens))]
                interaction_embedding += additional_padding

            interactions_embeddings.append(interaction_embedding)

        return interactions_embeddings

    def encode_labels(self, dataset, indexs):
        labels = []
        for interaction in dataset:
            label = interaction[3]
            labels.append([indexs['labels'][label]])

        return labels

    def build_tokens(self, analysis, entities, id_e1, id_e2):
        sentence_tokens = []

        # Store entities not in pair
        entities_other = entities.copy()
        del entities_other[id_e1]
        del entities_other[id_e2]

        # Adding pair entity information to analysis list
        analysis_list = [token for key, token in analysis['raw'].items() if token['word'] is not None]
        entity1 = entities[id_e1]
        analysis_list = self.map_entities(analysis_list, entity1, id_e1, 'DRUG1')
        entity2 = entities[id_e2]
        analysis_list = self.map_entities(analysis_list, entity2, id_e2, 'DRUG2')

        # Adding other entities information to analysis list
        for key, item in entities_other.items():
            analysis_list = self.map_entities(analysis_list, item, key, 'DRUG_OTHER')

        count_entity = 0
        for token in analysis_list:
            if 'entity_name' in token:
                if count_entity == 0:
                    count_entity += 1
                    sentence_tokens.append((f"<{token['entity_name']}>",
                                            f"<{token['entity_name']}>",
                                            f"<{token['entity_name']}>"))
            else:
                count_entity = 0
                sentence_tokens.append((token['word'], token['lemma'], token['tag']))
        return sentence_tokens

    def map_entities(self, analysis_list, entity, id, name):
        mapped_list = list(
            map(lambda x:
                dict(x, **({'entity_id': id, 'entity_name': name, 'entity_type': entity[1]}
                           if (x['start'] >= int(entity[0][0])) and (x['end'] <= int(entity[0][1])) else {})),
                analysis_list))

        return mapped_list

    @staticmethod
    def find_between_tags(lst, start_tag, end_tag):
        start_index = lst.index(start_tag)
        end_index = lst.index(end_tag, start_index)
        return lst[start_index: end_index + 1]




    @staticmethod
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")