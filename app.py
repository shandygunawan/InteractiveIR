from flask import Flask, render_template, request
from classes import Inputs, InvertedFile
import json
import helpers

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/result', methods=["POST"])
def result():
    helpers.set_settings(request.form)
    inputs = Inputs(request.form)

    inverted_file = InvertedFile(inputs.docs)
    return json.dumps(inverted_file.inverted_file)


if __name__ == '__main__':
    app.run()
