import os
from typing import List, Any, Union
from xml.dom.minidom import parse, parseString
from ..tokenizer import tokenizer
from eval import evaluator
from difflib import SequenceMatcher


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
            for i, line in enumerate(lines[:10000]):

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
        with open(path, encoding="utf8") as f:
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

    def brand_names(self, path):
        brandnames = []
        # Guardar los ngrams mas comunes que se corresponden a drugs
        with open(path) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                text = line.strip().lower()
                brandnames.append(text)

        return brandnames

    #https: // www.duffysrehab.com / resources / articles / schedule - of - drugs /
    def group_names(self, path):
        groupnames = []
        # Guardar los ngrams mas comunes que se corresponden a drugs
        with open(path) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line_list = line.split(':')
                text = line_list[0].strip().lower()
                groupnames.append(text)

        return groupnames

    def group_names_v2(self, path):
        groupnames = []
        # Guardar los ngrams mas comunes que se corresponden a drugs
        with open(path) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                text = line.lower()
                groupnames.append(text)

        return groupnames

    @staticmethod
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def hasNumbers(inputString):
        return any(char.isdigit() for char in inputString)

    def hasSpecialChars(self, inputString):
        return any(char in self.special_chars for char in inputString)

    @staticmethod
    def hasLetters(inputString):
        return any(c.isalpha() for c in inputString)

    @staticmethod
    def countCapitalized(inputString):
        return sum(1 for c in inputString if c.isupper())

utils = Utils()


class Parser(object):
    def __init__(self, path, out_path, resources_paths):
        self.path = path
        self.tokenizer = tokenizer()
        self.out_path = out_path
        self.brands = utils.brand_names(resources_paths.brandnames)
        self.groups = utils.group_names(resources_paths.groupnames)
        self.groups += utils.group_names_v2(resources_paths.groupnamesv2)
        #self.drugs = utils.ngrams_DrugBank(resources_paths.drugbank)
        #self.drugs += utils.ngrams_HSDB(resources_paths.hsdb)
        self.drugs = utils.drug_suffix(resources_paths.drugsuffix)
        self.drugs = list(set(self.drugs))
        #self.drugs=[]

    def extract_entities(self, tokens):
        entities = []
        for token in tokens:
            ngram, start, end = token
            ngram_lower = ngram.lower()

            classified = False
            if utils.countCapitalized(ngram) >= 3 and utils.countCapitalized(ngram) <= 4:
                classified = True
                entities.append({
                    "offset": f"{start}-{end}",
                    "text": ngram,
                    "type": "drug_n"
                })

            if utils.hasNumbers(ngram) and utils.hasSpecialChars(ngram) and len(ngram) > 5 and utils.hasLetters(ngram):
                classified = True
                entities.append({
                    "offset": f"{start}-{end}",
                    "text": ngram,
                    "type": "drug_n"
                })

            if not classified:
                for group in self.groups:
                    if ((ngram_lower in group) or (group in ngram_lower)) and len(ngram_lower) > 5:
                        classified = True
                        entities.append({
                            "offset": f"{start}-{end}",
                            "text": ngram,
                            "type": "group"
                        })

            if not classified:
                for drug in self.drugs:
                    if drug in ngram_lower:
                        classified = True
                        entities.append({
                            "offset": f"{start}-{end}",
                            "text": ngram,
                            "type": "drug"
                        })

            if not classified:
                for brand in self.brands:
                    if (((brand in ngram_lower)) and len(ngram_lower) > 5):
                        classified = True
                        entities.append({
                            "offset": f"{start}-{end}",
                            "text": ngram,
                            "type": "brand"
                        })

        return entities

    def parse_dir(self):
        with open(self.out_path, 'w') as outfile:
            for i, f in enumerate(os.listdir(self.path)):
                #print(f"File {i+1}/{len(os.listdir(self.path))}")
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

    def evaluate(self, path):
        evaluator.evaluate("NER", path, self.out_path)
