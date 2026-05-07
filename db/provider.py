from abc import ABC, abstractmethod


class DatabaseProvider(ABC):
    """Abstract interface for querying product data by barcode."""

    @abstractmethod
    def get_product(self, barcode: str) -> dict:
        """
        Fetch product information for a given barcode.
        Must return a dictionary with at least the key 'ingredients_text'.
        Example:
            {
                'product_name': 'Example Product',
                'ingredients_text': 'Water, sugar, ...'
            }
        Raises:
            ValueError: if the barcode is not found.
        """
        pass
