from nltk.tokenize import word_tokenize
import nltk
from nltk import ngrams
nltk.download('punkt')
from nltk.parse.corenlp import CoreNLPDependencyParser


class tokenizer(object):
    def __init__(self):
        self.word_tokenizer = word_tokenize

    @staticmethod
    def words_to_ngrams(words, n, sep=" "):
        return [sep.join(words[i:i + n]) for i in range(len(words) - n + 1)]

    def tokenize(self, stext, ngrams):
        words = self.word_tokenizer(stext)
        tokenized_sentence = []
        for i in range(ngrams):
            ngrams = self.words_to_ngrams(words, i+1)
            offset = 0
            for word in ngrams:
                crop_text = stext[offset:]
                start_indx = crop_text.find(word) + offset
                end_indx = start_indx + len(word) - 1

                tokenized_sentence.append((word, start_indx, end_indx))
                offset = end_indx + 1

        return tokenized_sentence

    def analyze(self, stext):
        myparser = CoreNLPDependencyParser(url="http://localhost:9000")
        mytree, = myparser.raw_parse(stext)
        #TODO: Add start-end token.
        pass