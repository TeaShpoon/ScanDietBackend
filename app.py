from db.provider import DatabaseProvider
from db.OpenFoodFacts import OpenFoodFactsProvider
from ner.predictor import NERPredictor
from ner.utils import reverse_ner, get_spans
from flask import Flask, jsonify
import sys

db_provider: DatabaseProvider = OpenFoodFactsProvider()

checkpoint_file = sys.argv[1]
model = NERPredictor(checkpoint_file)

app = Flask(__name__)


@app.route("/barcode/<code>")
def thething(code):
    p = db_provider.get_product(code)
    ingredients = p.get("ingredients_text", "No ingredients listed")
    labels = model.predict_sentence(ingredients)
    spans = get_spans(labels, ingredients)
    r = {"ingredients": ingredients, "labels": reverse_ner(spans)}
    return jsonify(r)


app.run()
