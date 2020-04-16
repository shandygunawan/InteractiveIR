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
            if word not in inverted_file.keys() or doc_id not in inverted_file[word]:
                vector_inverted.append(0)
            else:
                vector_inverted.append(inverted_file[word][doc_id])
            scanned_words.append(word)

    return int(np.dot(vector_query, vector_inverted))


def result_interactive(inverted_file, inputs):
    ir_result = {
        "result": []
    }

    if len(inputs.queries.keys()) > 1:
        return ir_result

    for doc_id in inputs.docs.keys():
        for query_id in inputs.queries.keys():
            ir_result["result"].append(
                {
                    "id": doc_id,
                    "document": inputs.docs_raw[doc_id],
                    "score": cosine_similarity(
                        inverted_file=inverted_file,
                        query=inputs.queries[query_id],
                        doc_id=doc_id)
                }
            )

    # Sort desc based on score
    ir_result['result'].sort(key=lambda x: x['score'], reverse=True)

    return ir_result


def result_experiment(inverted_file, inputs):
    ir_result = {
        "recall": -1,
        "precision": -1,
        "mean_average_precision": -1,
        "query_performances": {}
    }

    # Get cosine scores of all documents
    cosine_scores = []
    for query_id in inputs.queries.keys():
        for doc_id in inputs.docs.keys():
            cosine_scores.append(cosine_similarity(inverted_file=inverted_file, query=inputs.queries[query_id], doc_id=doc_id))

    # Get threshold quartile
    threshold = int(np.percentile(cosine_scores, 50))

    # Documents with cosine score > threshold will be marked as relevant
    relevance_found = []
    retrieved_count = []
    query_performances = {}
    for query_id in inputs.queries.keys():
        query_performances[query_id] = {}
        query_relevances = []
        query_retrieved = []
        for doc_id in inputs.docs.keys():
            if cosine_similarity(inverted_file=inverted_file, query=inputs.queries[query_id], doc_id=doc_id) > threshold:
                # For query's precision
                query_retrieved.append(doc_id)

                # For global precision
                if doc_id not in retrieved_count:
                    retrieved_count.append(doc_id)

                if doc_id in inputs.relevances[query_id]:
                    query_relevances.append(doc_id)
                    if doc_id not in relevance_found:
                        relevance_found.append(doc_id)

        # Query bby query Performance
        query_performances[query_id]['text'] = inputs.queries_raw[query_id]
        query_performances[query_id]['recall'] = len(query_relevances) / len(inputs.relevances[query_id])
        query_performances[query_id]['precision'] = len(query_relevances) / len(query_retrieved)

        # MAP : https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
        if len(query_relevances) > 0:
            avg_precision = 0
            true_positives = 0

            for index, doc_id in enumerate(query_retrieved):  # query_retrieved is predicted positives
                if doc_id in query_relevances:  # query_relevances is true positives
                    true_positives += 1
                    avg_precision += true_positives / index

            query_performances[query_id]['average_precision'] = avg_precision / len(query_relevances)
        else:
            query_performances[query_id]['average_precision'] = 0

    # For global recall
    relevances_total = []
    for query_id in inputs.relevances.keys():
        relevances_total.extend(inputs.relevances[query_id])
        for doc_id in inputs.relevances[query_id]:
            if doc_id not in relevances_total:
                relevances_total.append(doc_id)

    # Global recall
    if len(relevances_total) == 0 or len(relevance_found) == 0:
        ir_result['recall'] = 0
    else:
        ir_result['recall'] = len(relevance_found) / len(relevances_total)

    # Global precision
    ir_result['precision'] = len(relevance_found) / len(retrieved_count)

    # Global average precision (MAP)
    sum_ap = 0
    for query_id in query_performances:
        sum_ap += query_performances[query_id]['average_precision']

    ir_result['mean_average_precision'] = sum_ap / len(inputs.queries.keys())

    ir_result['query_performances'] = query_performances

    return ir_result
