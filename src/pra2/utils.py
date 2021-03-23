class Utils(object):
    def __init__(self, drugbank_path, hsdb_path):
        self.drugs = self.get_drugbank(drugbank_path)
        self.drugs['drug_n'] = self.get_hsdb(hsdb_path)

    def get_drugbank(self, path):
        drugs = {
            "group": [],
            "brand": [],
            "drug": []
        }
        with open(path, encoding="utf8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                text, type = line.strip().lower().split('|')
                drugs[type].append(text)

        return drugs

    def get_hsdb(self, path):
        drugs = []
        with open(path, encoding="utf8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                text = line.strip().lower()
                drugs.append(text)

        return drugs