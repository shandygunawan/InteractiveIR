import settings


def set_settings(req_form):
    settings.tf = req_form["TF"]
    settings.idf = req_form["IDF"]
    settings.stemming = req_form["Stemming"]
    settings.normalization = req_form["Normalization"]
    settings.stopwords = req_form["Stopwords"]
    settings.ir_type = req_form["Retrieval"]
