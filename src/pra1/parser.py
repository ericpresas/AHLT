import os
from xml.dom.minidom import parse, parseString
from ..tokenizer import tokenizer
from eval import evaluator


class Utils(object):
    def __init__(self):
        self.tokenizer = tokenizer()
        self.special_chars = "!@#$%^&*()-+?_=,<>/-[]"

    def ngrams_DrugBank(self, path):
        drugs_ngrams = []
        # Guardar los ngrams mas comunes que se corresponden a drugs
        with open(path, encoding="utf8") as f:
            lines = f.readlines()
            ngrams_drugs = []
            counts = []
            for i, line in enumerate(lines[:1000]):

                text, type = line.strip().lower().split('|')
                if type == 'drug':  # Solo miramos si es tipo drug
                    tokens = self.tokenizer.words_to_ngrams(text, 5, sep='')  # Extraemos ngramas de longitud 5 letras
                    for token in tokens:  # Para cada ngram
                        if ' ' not in token:  # No nos interesan los que contengan un espacio
                            if not any(c in self.special_chars for c in token):
                                if token in ngrams_drugs:  # Si esta en la lista, sumamos al contador
                                    indx_token = ngrams_drugs.index(token)
                                    counts[indx_token] += 1
                                else:  # Si no esta en la lista lo añadimos
                                    ngrams_drugs.append(token)
                                    counts.append(1)

                    if i % 50 == 0:
                        print(
                            f"{i + 1}/{len(lines)} analyzed. List size: {len(ngrams_drugs)}, Max count: {max(counts)}")

            # Ordenamos de mayor a menor frequencia de aparicion
            indx_max_ngrams = [i[0] for i in sorted(enumerate(counts), key=lambda x: x[1], reverse=True)]

            # Nos quedamos con los primeros
            print('Common ngrams:')
            for indx in indx_max_ngrams[:200]:
                drugs_ngrams.append(ngrams_drugs[indx])
        return drugs_ngrams

    def ngrams_HSDB(self, path):
        drugs_ngrams = []
        # Guardar los ngrams mas comunes que se corresponden a drugs
        with open(path, encoding="utf8") as f:
            lines = f.readlines()
            ngrams_drugs = []
            counts = []
            for i, line in enumerate(lines[:1000]):

                text = line.strip().lower()
                tokens = self.tokenizer.words_to_ngrams(text, 5, sep='')  # Extraemos ngramas de longitud 5 letras
                for token in tokens:  # Para cada ngram
                    if ' ' not in token:  # No nos interesan los que contengan un espacio
                        if token in ngrams_drugs:  # Si esta en la lista, sumamos al contador
                            indx_token = ngrams_drugs.index(token)
                            counts[indx_token] += 1
                        else:  # Si no esta en la lista lo añadimos
                            ngrams_drugs.append(token)
                            counts.append(1)

                    if i % 50 == 0:
                        print(
                            f"{i + 1}/{len(lines)} analyzed. List size: {len(ngrams_drugs)}, Max count: {max(counts)}")

            # Ordenamos de mayor a menor frequencia de aparicion
            indx_max_ngrams = [i[0] for i in sorted(enumerate(counts), key=lambda x: x[1], reverse=True)]

            # Nos quedamos con los primeros
            print('Common ngrams:')
            for indx in indx_max_ngrams[:200]:
                drugs_ngrams.append(ngrams_drugs[indx])
        return drugs_ngrams

utils = Utils()


class Parser(object):
    def __init__(self, path, out_path):
        self.path = path
        self.tokenizer = tokenizer()
        self.out_path = out_path
        self.drugs = utils.ngrams_DrugBank('resources/DrugBank.txt')
        self.drugs += utils.ngrams_HSDB('resources/HSDB.txt')
        self.drugs = list(set(self.drugs))

    def extract_entities(self, tokens):
        entities = []
        for token in tokens:
            ngram, start, end = token
            # TODO: Rule-based model: buscar mas reglas
            for drug in self.drugs:
                if drug in ngram:
                    entities.append({
                        "offset": f"{start}-{end}",
                        "text": ngram,
                        "type": "drug"
                    })
        return entities

    def parse_dir(self):
        with open(self.out_path, 'w') as outfile:
            for f in os.listdir(self.path):
                tree = parse(f"{self.path}/{f}")
                sentences = tree.getElementsByTagName("sentence")
                for s in sentences:
                    sid = s.attributes["id"].value
                    stext = s.attributes["text"].value

                    # Tokenizamos por palabras ngrams=1
                    tokens = self.tokenizer.tokenize(stext, ngrams=1)

                    # Para cada sentence extraemos las palabras que son tipo drug
                    entities = self.extract_entities(tokens)

                    for e in entities:
                        print(sid + "|" + e["offset"] + "|" + e["text"] + "|" + e["type"], file=outfile)

    def path_features(self, output_file):
        with open(output_file, 'w') as outfile:
            # Process each file in directory
            for f in os.listdir(self.path):
                tree = parse(f"{self.path}/{f}")
                sentences = tree.getElementsByTagName("sentence")
                for s in sentences:
                    sid = s.attributes["id"].value
                    stext = s.attributes["text"].value

                    gold = []
                    entities = s.getElementsByTagName("entity")
                    for e in entities:
                        offset = e.attributes["charOffset"].value
                        (start, end) = offset.split(";")[0].split("-")
                        gold.append((int(start), int(end), e.attributes["type"].value))

                    tokens = self.tokenizer.tokenize(stext, ngrams=1)
                    features = self.feature_extractor.extract_features(tokens)

                    for i in range(0, len(tokens)):
                        tag = self.get_tag(tokens[i], gold)
                        print(sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t', file=outfile)

                    print(file=outfile)

    def evaluate(self):
        evaluator.evaluate("NER", 'data/devel', self.out_path)


class Parser2(object):
    def __init__(self, path, out_path):
        self.path = path
        self.tokenizer = tokenizer()
        self.out_path = out_path
        self.feature_extractor = FeatureExtractor()
        print('a')

    def path_features(self, output_file):
        with open(output_file, 'w') as outfile:
            # Process each file in directory
            for f in os.listdir(self.path):
                tree = parse(f"{self.path}/{f}")
                sentences = tree.getElementsByTagName("sentence")
                for s in sentences:
                    sid = s.attributes["id"].value
                    stext = s.attributes["text"].value

                    gold = []
                    entities = s.getElementsByTagName("entity")
                    for e in entities:
                        offset = e.attributes["charOffset"].value
                        (start, end) = offset.split(";")[0].split("-")
                        gold.append((int(start), int(end), e.attributes["type"].value))

                    tokens = self.tokenizer.tokenize(stext, ngrams=1)
                    features = self.feature_extractor.extract_features(tokens)

                    for i in range(0, len(tokens)):
                        tag = self.get_tag(tokens[i], gold)
                        print(sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t', file=outfile)

                    print(file=outfile)

