import os
from dotenv import load_dotenv
import pytest
from google import genai


# Load env and use GOOGLE_API_KEY; skip test if missing
load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
pytest.skip("GOOGLE_API_KEY not set; skipping external model listing test.", allow_module_level=True) if not api_key else None


def test_list_models():
    client = genai.Client(api_key=api_key)
    models = list(client.models.list())
    # Basic sanity: ensure we got a list-like response
    assert isinstance(models, list)
    # If the API returns at least one model, check it has a name
    if models:
        assert hasattr(models[0], "name")