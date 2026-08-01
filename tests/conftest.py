import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def num_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, None],
        "score": [80.5, 90.0, 70.3, 85.1, 60.0]
    })

@pytest.fixture
def qual_df():
    return pd.DataFrame({
        "city": ["Milano", "Roma", "Milano", "Torino", "Roma"],
        "category": ["A", "B", "A", "C", "B"]
    })

@pytest.fixture
def text_df():
    return pd.DataFrame({
        "review": ['The food was absolutely amazing! I loved the flavors and the presentation was beautiful.',
                    'I had a terrible experience at this restaurant. The service was slow and the food was cold.',
                    'The ambiance was perfect for a romantic dinner. The lighting and music created a cozy atmosphere.',
                    'I would highly recommend this place to anyone looking for a great dining experience. The staff was friendly and attentive, and the food was delicious.',
                    'The portions were small and overpriced. I expected more for the price I paid.']
    })



