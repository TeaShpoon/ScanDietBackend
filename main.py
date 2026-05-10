from db.provider import DatabaseProvider
from db.OpenFoodFacts import OpenFoodFactsProvider
from ner.predictor import NERPredictor
from ner.utils import reverse_ner, get_spans
from flask import Flask, jsonify, request
from sys import argv

db_provider: DatabaseProvider = OpenFoodFactsProvider()

if len(argv) < 1:
    model_file = "./model.pt"
else:
    model_file = argv[1]


model = NERPredictor(model_file)

app = Flask(__name__)


@app.route("/barcode/<code>")
def handle_barcode(code):
    product = db_provider.get_product(code)
    ingredients = product["ingredients_text"]
    name = product["product_name"]
    labels = model.predict_sentence(ingredients)
    spans = get_spans(labels, ingredients)
    response = {"name": name, "ingredients": ingredients, "labels": reverse_ner(spans)}
    return jsonify(response)


@app.route("/ingredients", methods=["POST"])
def handle_ingredients():
    data = request.get_data(as_text=True)
    labels = model.predict_sentence(data)
    spans = get_spans(labels, data)
    response = {"ingredients": data, "labels": reverse_ner(spans)}
    return jsonify(response)


app.run(host="0.0.0.0", port=3000, debug=True)
