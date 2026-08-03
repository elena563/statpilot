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
        "age": [25, 30, 35, 40, None, 28, 32, 38, 45, 50, 55, 60, 65, 70, 75],
        "score": [80.5, 90.0, 70.3, 85.1, 60.0, 75.0, 95.2, 88.8, 92.5, 78.9, 82.0, 91.5, 76.2, 89.0, 74.5]
    })

@pytest.fixture
def qual_df():
    return pd.DataFrame({
        "city": ["Milano", "Roma", "Milano", "Torino", "Roma", "Torino", "Milano", "Roma", "Torino", "Milano", "Roma", "Torino", "Milano", "Roma", "Torino"],
        "category": ["A", "B", "A", "C", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"]
    })

@pytest.fixture
def text_df():
    return pd.DataFrame({
        "review": ['The food was absolutely amazing! I loved the flavors and the presentation was beautiful.',
                    'I had a terrible experience at this restaurant. The service was slow and the food was cold.',
                    'The ambiance was perfect for a romantic dinner. The lighting and music created a cozy atmosphere.',
                    'I would highly recommend this place to anyone looking for a great dining experience. The staff was friendly and attentive, and the food was delicious.',
                    'The portions were small and overpriced. I expected more for the price I paid.',
                    'Horrible', 'The food was cold and tasteless. I will not be coming back.',
                    'The service was exceptional. The waitstaff was attentive and knowledgeable about the menu.',
                    'I had a wonderful experience at this restaurant. The food was flavorful and the presentation was beautiful.',
                    'The restaurant had a great atmosphere. The lighting and music created a cozy and inviting environment.',
                    "The food was absolutely amazing! I loved the flavors and the presentation was beautiful.",
                    "I had a terrible experience at this restaurant. The service was slow and the food was cold.",
                    "The ambiance was perfect for a romantic dinner. The lighting and music created a cozy atmosphere.",
                    "I would highly recommend this place to anyone looking for a great dining experience. The staff was friendly and attentive, and the food was delicious.",
                    "The portions were small and overpriced. I expected more for the price I paid."]
    })

@pytest.fixture
def full_df(num_df, qual_df, text_df):
    return pd.concat([num_df, qual_df, text_df], axis=1)
