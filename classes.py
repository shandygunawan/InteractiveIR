from collections import defaultdict
from math import log2
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


class Docs:
    docs = {}
    settings = {}

    def __init__(self, documents, param_settings):
        self.settings = param_settings
        for doc_id in documents.keys():
            self.add_update(doc_id, documents[doc_id])

    def get_settings(self):
        return self.settings

    def get_all(self):
        return self.docs

    def get_id(self, doc_id):
        if doc_id in self.docs:
            return self.docs[doc_id]
        else:
            return -1

    def add_update(self, doc_id, doc):
        self.docs[doc_id] = self.preprocessing(doc)

    def preprocessing(self, doc):
        stop_words = set(stopwords.words('english'))

        # Case Folding
        words = doc['document'].lower()

        # Tokenize
        words = word_tokenize(words)

        # Stopwords Removal
        if self.settings['stopwords'] == 'true':
            words = [w for w in words if not w in stop_words]

        # Stemming (Porter)
        if self.settings['stemming'] == 'true':
            ps = PorterStemmer()
            words = [ps.stem(w) for w in words]

        doc['document'] = words

        return doc


class TFIDF:

    idf_dict = {}
    docs = {}
    tf_type = ""

    def __init__(self, param_docs, tf):
        self.docs = param_docs
        self.tf_type = tf

    #
    # IDF
    #
    def is_idf_exist(self, term):
        if term in self.idf_dict:
            return True
        else:
            return False

    def get_idf(self, term):
        if term in self.idf_dict:
            return self.idf_dict[term]
        else:
            return -1

    def calculate_idf(self, term):
        # Get number of documents that have term inside them
        df_count = 1
        for doc_id in self.docs.keys():
            if term in self.docs[doc_id]['document']:
                df_count += 1

        idf = len(self.docs) / df_count
        self.idf_dict[term] = idf
        return idf

    #
    # TF
    #
    def get_tf_type(self):
        return self.tf_type

    def calculate_tf(self, term, doc_id):
        if self.tf_type == "raw":
            return self.docs[doc_id]['document'].count(term)
        elif self.tf_type == "binary":
            if term in self.docs[doc_id]['document']:
                return 1
            else:
                return 0
        elif self.tf_type == "log":
            if term not in self.docs[doc_id]['document']:
                return 0
            else:
                return 1 + log2(self.docs[doc_id]['document'].count(term))
        elif self.tf_type == "aug":
            count_term = self.docs[doc_id]['document'].count(term)

            # Get the number of occurrences for the item with the highest occurrences
            d = defaultdict(int)
            for i in self.docs[doc_id]['document']:
                d[i] += 1
            count_max = (max(d.items(), key=lambda x: x[1]))[1]

            return 0.5 + (0.5 * (count_term / count_max))
        else:
            return -1


