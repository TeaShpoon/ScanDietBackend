from db.provider import DatabaseProvider
from db.OpenFoodFacts import OpenFoodFactsProvider
from ner.predictor import NERPredictor
from ner.utils import reverse_ner, get_spans
from flask import Flask, jsonify, request
import argparse

db_provider: DatabaseProvider = OpenFoodFactsProvider()

parser = argparse.ArgumentParser(description="Run a model on a specified device.")
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Device to run inference on",
)
parser.add_argument(
    "model_path",
    type=str,
    default="./model.pt",
    help="Path to the model file",
)

parser.add_argument(
    "--port",
    type=int,
    default=3000,
    help="Port number to listen on (default: 3000)",
)

parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Enable debug mode (default: False)",
)

parser.add_argument(
    "--host",
    type=str,
    default="localhost",
    help="Host address to bind to (default: localhost)",
)

args = parser.parse_args()

model = NERPredictor(args.model_path, device=args.device)

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


app.run(host=args.host, port=args.port, debug=args.debug)
