from datetime import datetime
import pandas as pd
from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS

from modelling import fetch_preds

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/get_chart")
def get_chart():
    print(request)
    models = request.args.get("models").lower()
    if not models:
        return jsonify({"error": "Model parameter is required"}), 400

    models = models.lower().split(",")
    start_date_str = request.args.get("start_date")
    end_date_str = request.args.get("end_date")

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    base_date = datetime.strptime("2007-09-01", "%Y-%m-%d").date()

    start_idx = (start_date - base_date).days
    end_idx = (end_date - base_date).days

    response = {"ili": None}
    for model in models:
        try:
            preds, ili = fetch_preds(model, start_idx, end_idx)
            if response["ili"] is None:
                response["ili"] = ili
            response[model] = preds

        except Exception as e:
            print({"error": str(e)})
            return jsonify({"error": str(e)}), 500

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
