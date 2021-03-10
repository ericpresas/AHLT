

class FeatureExtractor(object):
    def __init__(self):
        pass

    def extract_features(self, tokens):
        features = []
        for i, token in enumerate(tokens):
            ngram, start, end = token
            form = ngram
            suf4 = ngram[-4:] if len(ngram) >= 4 else ngram
            prev_index = i - 1
            next_index = i + 1
            prev_ngram = ''
            if prev_index >= 0:
                prev_ngram, _, _ = tokens[prev_index]
            else:
                prev_ngram = '_BoS_'

            next_ngram = ''
            if next_index < (len(tokens) - 1):
                next_ngram, _, _ = tokens[next_index]
            else:
                next_ngram = '_EoS_'

            token_features = [
                f"form={ngram}",
                f"suf4={suf4}",
                f"next={next_ngram}",
                f"prev={prev_ngram}"
            ]
            features.append(token_features)

        return features

    def get_tag(self, token, gold):
        text, start, end = token
        pos = None
        type_pos = None
        for entity in gold:
            entity_start, entity_end, type = entity
            if start >=entity_start and start<=entity_end:
                pos = 'B' if entity_start==start else 'I'
                type_pos = type
        if pos is None:
            pos = 'O'

        if type_pos is None:
            result = 'O'
        else:
            result = f"{pos}-{type_pos}"
        return result

