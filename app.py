from flask import Flask, render_template, request
from classes import Inputs, InvertedFile
import json
import helpers
import settings

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/result', methods=["POST"])
def result():
    # Parse inputs (documents, queries, relevances)
    helpers.set_settings(request.form)
    inputs = Inputs(request.form)

    # Create inverted file
    inverted_file = InvertedFile(inputs.docs)

    # Execute IR System (Interactive or experiment)
    if settings.ir_type == "interactive":
        ir_result = helpers.result_interactive(
           inverted_file=inverted_file.inverted_file,
           inputs=inputs
        )

        return render_template("result_interactive.html",
                               ir_result=ir_result)
    else:  # Experiment
        ir_result = helpers.result_experiment(
            inverted_file=inverted_file.inverted_file,
            inputs=inputs
        )
        return render_template("result_experiment.html",
                               ir_result=ir_result,
                               inverted_file=inverted_file.inverted_file)


if __name__ == '__main__':
    app.run()
