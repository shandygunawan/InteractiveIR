from flask import Flask, render_template, request
from classes import Docs
import json
import helpers

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/result', methods=["POST"])
def result():
    settings = helpers.get_settings(request.form)
    inputs = helpers.get_inputs(settings['type'], request.form)
    docs = Docs(inputs['documents'], settings)

    inverted_file = helpers.create_inverted_file(docs, settings)
    return json.dumps(inverted_file)


if __name__ == '__main__':
    app.run()
