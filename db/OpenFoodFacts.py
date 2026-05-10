import requests
from db.provider import DatabaseProvider


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
        name = (
            product.get("product_name_ru") or product.get("product_name") or "Unknown"
        )

        return {
            "product_name": name,
            "ingredients_text": product.get("ingredients_text", ""),
        }
