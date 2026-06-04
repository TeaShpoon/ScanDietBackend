import requests
from db.provider import DatabaseProvider
import html


class OpenFoodFactsProvider(DatabaseProvider):
    BASE_URL = "https://world.openfoodfacts.org/api/v2/product/"
    USER_AGENT = "ScanDiet/1.0"

    def get_product(self, barcode: str) -> dict:

        url = f"{self.BASE_URL}{barcode}.json"
        headers = {"User-Agent": self.USER_AGENT}

        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()

        data = response.json()

        if data.get("status") != 1:
            raise ValueError(f"Product with barcode {barcode} not found.")

        product = data["product"]
        ingredients = full_unescape(product.get("ingredients_text", ""))

        name = full_unescape(
            (product.get("product_name_ru") or product.get("product_name") or "Unknown")
        )

        return {
            "product_name": name,
            "ingredients_text": ingredients,
        }


def full_unescape(s):
    prev = None
    while prev != s:
        prev = s
        s = html.unescape(s)
    return s
