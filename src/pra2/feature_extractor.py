from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
from difflib import SequenceMatcher

class FeatureExtractor(object):
    def __init__(self, utils):
        self.special_chars = "!@#$%^&*()-+?_=,<>/-[]:"
        self.utils = utils

    @staticmethod
    def sentiment_scores(sentence):

        # Create a SentimentIntensityAnalyzer object.
        sid_obj = SentimentIntensityAnalyzer()

        # polarity_scores method of SentimentIntensityAnalyzer
        # oject gives a sentiment dictionary.
        # which contains pos, neg, neu, and compound scores.
        sentiment_dict = sid_obj.polarity_scores(sentence)
        if sentiment_dict['compound'] >= 0.05:
            return "Positive"

        elif sentiment_dict['compound'] <= - 0.05:
            return "Negative"

        else:
            return "Neutral"

    @staticmethod
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    def get_aditional_resorces(self, ngram):
        categories = ['drug', 'brand', 'group', 'drug_n']
        similarities = {}
        for category in categories:
            similarities[category] = 0
            tmp = []
            if ngram.lower() in self.utils.drugs[category]:
                tmp.append(ngram)

            similarities[category] = len(tmp)

        return similarities


    def extract_features(self, tokens, stext):
        features = []
        for i, token in enumerate(tokens):
            ngram, start, end = token
            form = ngram

            # Gets if the ngram is present for each category (binary)
            resources = self.get_aditional_resorces(ngram)
            drug_counts = resources['drug']
            group_counts = resources['group']
            brand_counts = resources['brand']
            drug_n_counts = resources['drug_n']

            # Extract different prefixes and suffixes from the ngram
            suf4 = ngram[-4:] if len(ngram) >= 4 else ngram
            suf3 = ngram[-3:] if len(ngram) >= 3 else ngram
            suf2 = ngram[-2:] if len(ngram) >= 2 else ngram
            pref4 = ngram[:4] if len(ngram) >= 4 else ngram
            pref3 = ngram[:3] if len(ngram) >= 3 else ngram
            pref2 = ngram[:2] if len(ngram) >= 2 else ngram
            prev_index = i - 1
            next_index = i + 1
            prev_ngram = ''

            # Extracts the previous word (if it's a special character will take the previous from previous)
            if prev_index >= 0:
                prev_ngram, _, _ = tokens[prev_index]
                if prev_ngram in self.special_chars:
                    if prev_index - 1 >= 0:
                        prev_ngram, _, _ = tokens[prev_index - 1]
            else:  # Beggining of the sentence if there is no previous words
                prev_ngram = '_BoS_'

            next_ngram = ''

            # Extracts the next word (if it's a special character will take the next from next)
            if next_index < (len(tokens) - 1):
                next_ngram, _, _ = tokens[next_index]
                if next_ngram in self.special_chars:
                    if next_index + 1 < (len(tokens) - 1):
                        next_ngram, _, _ = tokens[next_index + 1]
            else:  # End of the sentence if there is no next words
                next_ngram = '_EoS_'

            # Building the feature vector
            token_features = [
                f"form={ngram}",
                f"form-lower={ngram.lower()}",
                f"form-isupper={ngram.isupper()}",
                f"form-istitle={ngram.istitle()}",
                f"form-isdigit={ngram.isdigit()}",
                f"suf4={suf4}",
                f"suf3={suf3}",
                f"suf2={suf2}",
                f"pref4={pref4}",
                f"pref3={pref3}",
                f"pref2={pref2}",
                f"next={next_ngram}",
                f"next-lower={next_ngram.lower()}",
                f"prev={prev_ngram}",
                f"prev-lower={prev_ngram.lower()}",
                f"drug={drug_counts}",
                f"group={group_counts}",
                f"brand={brand_counts}",
                f"drug-n={drug_n_counts}"
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

