import pytest
import os
from service import *
from src.prediction_backend import *

#Creating a test client to use in the path testing
@pytest.fixture(name="client")
def fixture_client():
    service.config['TESTING'] = True
    client = service.test_client()
    print("Client up!")
    return client

#Getting a list of all models to use in backend testing
@pytest.fixture(name="models")
def fixture_models():
    return  Models