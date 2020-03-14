import settings
import numpy as np


def set_settings(req_form):
    settings.tf = req_form["TF"]
    settings.idf = req_form["IDF"]
    settings.stemming = req_form["Stemming"]
    settings.normalization = req_form["Normalization"]
    settings.stopwords = req_form["Stopwords"]
    settings.ir_type = req_form["Retrieval"]


def cosine_similarity(inverted_file, query, doc_id):
    vector_query = []
    vector_inverted = []
    scanned_words = []

    for word in query:
        if word not in scanned_words:
            vector_query.append(query.count(word))
            if doc_id not in inverted_file[word]:
                vector_inverted.append(0)
            else:
                vector_inverted.append(inverted_file[word][doc_id])
            scanned_words.append(word)

    return int(np.dot(vector_query, vector_inverted))


def result_interactive(inverted_file, inputs):
    ir_result = {
        "result": []
    }

    for idx, query in enumerate(inputs.queries):
        for doc_id in inputs.docs.keys():
            ir_result["result"].append(
                {
                    "id": doc_id,
                    "document": inputs.docs_raw[doc_id]['document'],
                    "score": cosine_similarity(inverted_file=inverted_file,
                                                query=query,
                                                doc_id=doc_id)
                }
            )

    # Sort desc based on score
    ir_result['result'].sort(key=lambda x: x['score'], reverse=True)

    return ir_result
