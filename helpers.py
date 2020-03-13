from classes import TFIDF
import json


def get_settings(req_form):
    return {
        "tf": req_form["TF"],
        "idf": req_form["IDF"],
        "stemming": req_form["Stemming"],
        "normalization": req_form['Normalization'],
        "stopwords": req_form['Stopwords'],
        "type": req_form['Retrieval']
    }


def get_inputs(ir_type, req_form):
    documents = json.loads(req_form['input_doc'])
    # queries = json.loads(req_form['input_query'])
    if ir_type == 'experiment':
        return {
            "documents": documents,
            "queries": req_form['input_query'],
            "relevance": req_form['input_relevance']
        }
    else:
        return {
            "documents": documents,
            "queries": req_form['input_query']
        }


def create_inverted_file(docs, settings):
    inverted_file = {}

    docs_dict = docs.get_all()
    tfidf = TFIDF(param_docs=docs_dict, tf=settings['tf'])

    for doc_id in docs_dict.keys():

        # Weighting
        for word in docs_dict[doc_id]['document']:
            if word not in inverted_file:
                inverted_file[word] = {}

            if doc_id not in inverted_file[word]:
                if settings['tf'] != 'none':
                    tf = tfidf.calculate_tf(word, doc_id)
                else:
                    tf = 1

                if settings['idf'] == 'true':
                    if tfidf.is_idf_exist(word):
                        idf = tfidf.get_idf(word)
                        print("idf fetched")
                    else:
                        idf = tfidf.calculate_idf(word)
                else:
                    idf = 1

                inverted_file[word][doc_id] = tf * idf

    return inverted_file
