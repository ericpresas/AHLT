import os
from xml.dom.minidom import parse
from ..tokenizer import tokenizer
from eval import evaluator


class Parser(object):
    def __init__(self, path, out_path):
        self.path = path
        self.tokenizer = tokenizer()
        self.out_path = out_path

    def path_process(self):
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

                    analysis = self.tokenizer.analyze(stext)

                    pairs = s.getElementsByTagName('pair')
                    for p in pairs:
                        id_e1 = p.attributes['e1'].value
                        id_e2 = p.attributes['e2'].value
                        ddi_type = self.check_interaction(analysis, entities, id_e1, id_e2)
                        if ddi_type != None:
                            print(sid + "|" + id_e1 + "|" + id_e2 + "|" + ddi_type, file=outfile)

    def check_interaction(self, analysis, entities, id_e1, id_e2):
        # TODO: check interaction
        return None

    def evaluate(self, path):
        evaluator.evaluate("DDI", path, self.out_path)








