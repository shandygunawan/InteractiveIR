from collections import defaultdict
from math import log2
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import settings
import json


class Inputs:
    # Constants
    TYPE_DOCUMENT = "document"
    TYPE_QUERY = "query"
    TYPE_RELEVANCE = "relevance"

    docs = {}
    queries = {}
    relevances = {}

    def __init__(self, req_form):
        json_docs = json.loads(req_form["input_doc"])
        self.set_docs(json_docs)

    #
    # DOCS
    #
    def set_docs(self, json_docs):
        for doc_id in json_docs.keys():
            self.add_doc(doc_id, json_docs[doc_id])

    def get_docs(self):
        return self.docs

    def add_doc(self, doc_id, doc):
        self.docs[doc_id] = self.preprocessing(input_type=self.TYPE_DOCUMENT,
                                               input_content=doc)

    #
    # QUERIES
    #
    def set_queries(self, json_queries):
        for query_id in json_queries.keys():
            self.add_query(query_id, json_queries[query_id])

    def add_query(self, query_id, query):
        self.queries[query_id] = self.preprocessing(input_type=self.TYPE_QUERY,
                                                    input_content=query)

    #
    # OTHER
    #
    def preprocessing(self, input_type, input_content):
        stop_words = set(stopwords.words('english'))

        # Case Folding
        if input_type == self.TYPE_DOCUMENT:
            words = input_content['document'].lower()
        else:
            words = input_content['document'].lower()

        # Tokenize
        words = word_tokenize(words)

        # Stopwords Removal
        if settings.stopwords == 'true':
            words = [w for w in words if not w in stop_words]

        # Stemming (Porter)
        if settings.stemming == 'true':
            ps = PorterStemmer()
            words = [ps.stem(w) for w in words]

        if input_type == self.TYPE_DOCUMENT:
            input_content['document'] = words
        else:
            input_content['document'] = words

        return input_content


class TFIDF:

    idf_dict = {}
    docs = {}

    def __init__(self, param_docs):
        self.docs = param_docs

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

    def calculate_tf(self, term, doc_id):
        if settings.tf == "raw":
            return self.docs[doc_id]['document'].count(term)
        elif settings.tf == "binary":
            if term in self.docs[doc_id]['document']:
                return 1
            else:
                return 0
        elif settings.tf == "log":
            if term not in self.docs[doc_id]['document']:
                return 0
            else:
                return 1 + log2(self.docs[doc_id]['document'].count(term))
        elif settings.tf == "aug":
            count_term = self.docs[doc_id]['document'].count(term)

            # Get the number of occurrences for the item with the highest occurrences
            d = defaultdict(int)
            for i in self.docs[doc_id]['document']:
                d[i] += 1
            count_max = (max(d.items(), key=lambda x: x[1]))[1]

            return 0.5 + (0.5 * (count_term / count_max))
        else:
            return -1


class InvertedFile:
    inverted_file = {}

    def __init__(self, docs):
        self.create(docs)

    def create(self, docs):
        self.inverted_file = {}

        tfidf = TFIDF(param_docs=docs)

        for doc_id in docs.keys():
            # Weighting
            for word in docs[doc_id]['document']:
                if word not in self.inverted_file:
                    self.inverted_file[word] = {}

                if doc_id not in self.inverted_file[word]:
                    if settings.tf != 'none':
                        tf = tfidf.calculate_tf(word, doc_id)
                    else:
                        tf = 1

                    if settings.idf == 'true':
                        if tfidf.is_idf_exist(word):
                            idf = tfidf.get_idf(word)
                        else:
                            idf = tfidf.calculate_idf(word)
                    else:
                        idf = 1

                    self.inverted_file[word][doc_id] = tf * idf
