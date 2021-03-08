import os
from typing import List, Any, Union
from xml.dom.minidom import parse, parseString
from .tokenizer import tokenizer
from eval import evaluator


class Utils(object):
    def __init__(self):
        self.tokenizer = tokenizer()
        self.special_chars = "!@#$%^&*()-+?_=,<>/-[]:"

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
                    #tokens = tokens[len(tokens) - 1:]
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
            for indx in indx_max_ngrams[:300]:
                drugs_ngrams.append(ngrams_drugs[indx])
        return drugs_ngrams

    def ngrams_HSDB(self, path):
        drugs_ngrams: List[Union[Union[str, List[Union[str, Any]]], Any]] = []
        # Guardar los ngrams mas comunes que se corresponden a drugs
        with open(path) as f:
            lines = f.readlines()
            ngrams_drugs = []
            counts = []
            for i, line in enumerate(lines[:1000]):

                text = line.strip().lower()
                tokens = self.tokenizer.words_to_ngrams(text, 5, sep='')  # Extraemos ngramas de longitud 5 letras
                tokens = tokens[len(tokens)-1:]
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
            for indx in indx_max_ngrams[:300]:
                drugs_ngrams.append(ngrams_drugs[indx])
        return drugs_ngrams
    def drug_suffix(self, path):
        drugs_ngrams: List[Union[Union[str, List[Union[str, Any]]], Any]] = []
        # Guardar los ngrams mas comunes que se corresponden a drugs
        with open(path) as f:
            lines = f.readlines()
            ngrams_drugs = []
            counts = []
            for i, line in enumerate(lines):

                text = line.strip().lower()
                token = text
                if ' ' not in token:
                    if not any(c in self.special_chars for c in token):
                        if len(token) > 4:
                            drugs_ngrams.append(token)

        return drugs_ngrams
utils = Utils()


class Parser(object):
    def __init__(self, path, out_path):
        self.path = path
        self.tokenizer = tokenizer()
        self.out_path = out_path
        self.drugs = utils.ngrams_DrugBank('E:\\UNI\\MASTER 1\\AHLT\\Session1\\Code\\resources\\DrugBank.txt')
        self.drugs += utils.ngrams_HSDB('E:\\UNI\\MASTER 1\\AHLT\\Session1\\Code\\resources\\HSDB.txt')
        self.drugs += utils.drug_suffix('E:\\UNI\\MASTER 1\\AHLT\\Session1\\Code\\resources\\drugSuffix.txt')
        self.drugs = list(set(self.drugs))
        #self.drugs=[]

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

    def evaluate(self):
        evaluator.evaluate("NER", 'data/devel', self.out_path)

