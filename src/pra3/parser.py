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

                    try:
                        analysis = self.tokenizer.analyze(stext)

                        pairs = s.getElementsByTagName('pair')
                        for p in pairs:
                            id_e1 = p.attributes['e1'].value
                            id_e2 = p.attributes['e2'].value
                            ddi = self.str2bool(p.attributes['ddi'].value)
                            if ddi:
                                ddi_type_gt = p.attributes['type'].value

                                if ddi_type_gt == 'effect':
                                    print(ddi_type_gt)
                            ddi_type = self.check_interaction(analysis, entities, id_e1, id_e2)
                            if ddi_type != None:
                                print(sid + "|" + id_e1 + "|" + id_e2 + "|" + ddi_type, file=outfile)
                    except Exception as e:
                        pass

    def check_interaction(self, analysis, entities, id_e1, id_e2):
        # TODO: check interaction
        print(f"Drug1: {entities[id_e1]}")
        print(f"Drug2: {entities[id_e2]}")
        print_sentence = []
        print_tag = []
        ddi_type = None
        for key, obj in analysis.items():
            if 'start' in obj:
                if obj['start'] >= int(entities[id_e1][0]) and obj['end'] <= int(entities[id_e2][1]):
                    #print(f"{key} - {obj}")
                    print_sentence.append(obj['word'])
                    print_tag.append(obj['tag'])
        print(" ".join(print_sentence))
        print(" ".join(print_tag))

        if 'MD' in print_tag:
            indx_md = print_tag.index('MD')
            if print_tag[indx_md + 1] == 'VB':
                ddi_type = 'effect'
                print(f"Found {ddi_type}")

        if 'FW' in print_tag:
            ddi_type = 'effect'
            print(f'Found {ddi_type}')
        return ddi_type

    @staticmethod
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    def evaluate(self, path):
        evaluator.evaluate("DDI", path, self.out_path)








