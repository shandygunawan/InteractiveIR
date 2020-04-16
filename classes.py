from collections import defaultdict
from copy import deepcopy
from math import log2
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import inflect
import re
import settings


class Inputs:
    # Constants
    TYPE_DOCUMENT = "document"
    TYPE_QUERY = "query"
    TYPE_RELEVANCE = "relevance"

    docs_raw = {}
    queries_raw = {}
    relevances_raw = {}

    docs = {}
    queries = {}
    relevances = {}

    def __init__(self, req_form):
        self.parse_text(
            input_type=self.TYPE_DOCUMENT,
            input_text=req_form["input_doc"])
        self.parse_text(
            input_type=self.TYPE_QUERY,
            input_text=req_form["input_query"])

        self.preprocess_text(input_type=self.TYPE_DOCUMENT)
        self.preprocess_text(input_type=self.TYPE_QUERY)

        if settings.ir_type == 'experiment':
            self.parse_relevances(req_form["input_relevance"])

    #
    # TEXT(DOCS & QUERIES)
    #
    def parse_text(self, input_type, input_text):
        lines = input_text.splitlines()
        text_detected = False
        text_id = ""
        text_content = ""

        if input_type == self.TYPE_DOCUMENT:
            self.docs_raw = {}
            self.docs = {}
        else:
            self.queries_raw = {}
            self.queries = {}

        for line in lines:
            # Start recording when a numeric string line detected
            if not text_detected:
                if line.isnumeric():
                    text_detected = True
                    text_id = line
            else:
                # End of document detected
                if line == '/':
                    text_detected = False

                    # Remove \r\n and whitespace on string's end
                    text_content = text_content.rstrip()
                    if input_type == self.TYPE_DOCUMENT:
                        self.docs_raw[text_id] = text_content
                    else:
                        self.queries_raw[text_id] = text_content

                    text_id = ""
                    text_content = ""

                # Keep appending line string as document's content
                else:
                    text_content += line
                    text_content += " "

    def preprocess_text(self, input_type):
        if input_type == self.TYPE_DOCUMENT:
            for key in self.docs_raw.keys():
                self.docs[key] = self.preprocessing(
                    input_content=self.docs_raw[key]
                )
        else:
            for key in self.queries_raw.keys():
                self.queries[key] = self.preprocessing(
                    input_content=self.queries_raw[key]
                )

    #
    # RELEVANCES
    #
    def parse_relevances(self, input_rel):
        lines = input_rel.splitlines()
        rel_detected = False
        rel_id = ""
        rel_content = []

        for line in lines:
            if not rel_detected:
                rel_id = line
                rel_detected = True
            else:
                # End of rel
                if line == '/':
                    rel_detected = False

                    self.relevances[rel_id] = rel_content
                    rel_id = ""
                    rel_content = []
                else:
                    line_splitted = line.split()
                    for doc_id in line_splitted:
                        rel_content.append(doc_id)

    #
    # OTHER
    #
    def preprocessing(self, input_content):
        stop_words = set(stopwords.words('english'))

        # Tokenize
        words = word_tokenize(input_content)

        # Normalization
        if settings.normalization == 'true':
            new_words = []
            p = inflect.engine()

            for word in words:
                new_word = deepcopy(word)
                # Replace numbers
                if new_word.isdigit():
                    new_word = p.number_to_words(new_word)

                # Remove punctuations
                new_word = re.sub(r'[^\w\s]', '', word)

                # Case-folding
                new_word = new_word.lower()

                new_words.append(new_word)

            words = new_words

        # Stopwords Removal
        if settings.stopwords == 'true':
            words = [w for w in words if not w in stop_words]

        # Stemming (Porter)
        if settings.stemming == 'true':
            ps = PorterStemmer()
            words = [ps.stem(w) for w in words]

        return words


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
        for text_id in self.docs.keys():
            if term in self.docs[text_id]:
                df_count += 1

        idf = len(self.docs) / df_count
        self.idf_dict[term] = idf
        return idf

    #
    # TF
    #
    def calculate_tf(self, term, text_id):
        if settings.tf == "raw":
            return self.docs[text_id].count(term)
        elif settings.tf == "binary":
            if term in self.docs[text_id]:
                return 1
            else:
                return 0
        elif settings.tf == "log":
            if term not in self.docs[text_id]:
                return 0
            else:
                return 1 + log2(self.docs[text_id].count(term))
        elif settings.tf == "aug":
            count_term = self.docs[text_id].count(term)

            # Get the number of occurrences for the item with the highest occurrences
            d = defaultdict(int)
            for i in self.docs[text_id]:
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

        for text_id in docs.keys():
            # Weighting
            for word in docs[text_id]:
                if word not in self.inverted_file:
                    self.inverted_file[word] = {}

                if text_id not in self.inverted_file[word]:
                    if settings.tf != 'none':
                        tf = tfidf.calculate_tf(word, text_id)
                    else:
                        tf = 1

                    if settings.idf == 'true':
                        if tfidf.is_idf_exist(word):
                            idf = tfidf.get_idf(word)
                        else:
                            idf = tfidf.calculate_idf(word)
                    else:
                        idf = 1

                    self.inverted_file[word][text_id] = tf * idf
