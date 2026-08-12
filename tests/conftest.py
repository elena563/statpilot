import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# 15 samples dfs
@pytest.fixture
def base_df():
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, None, 28, 32, 38, None, 45, 50, 55, 60, 65, 70, 75, 40, 28],
            "score": [80.5, 90.0, 70.3, 85.1, 60.0, 75.0, 95.2, 88.8, 92.5, 78.9, 82.0, 91.5, 76.2, 89.0, 74.5, 70.3, 85.1, 60.0],
            "city": ["Milano", "Roma", "Milano", "Torino", "Roma", "Torino", "Milano", "Roma", "Torino", "Milano", "Roma", "Torino", "Milano", "Roma", "Torino", "Milano", "Roma", "Torino"],
            "category": ["A", "B", "A", "C", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "review": [
                "The food was absolutely amazing! I loved the flavors and the presentation was beautiful.",
                "I had a terrible experience at this restaurant. The service was slow and the food was cold.",
                "The ambiance was perfect for a romantic dinner. The lighting and music created a cozy atmosphere.",
                "I would highly recommend this place to anyone looking for a great dining experience. The staff was friendly and attentive, and the food was delicious.",
                "The portions were small and overpriced. I expected more for the price I paid.",
                "Horrible",
                "The food was cold and tasteless. I will not be coming back.",
                "The food was okay, nothing special.",
                "The service was slow and the staff seemed disinterested.",
                "The restaurant was clean and well-maintained. The staff was friendly and accommodating.",
                "The service was exceptional. The waitstaff was attentive and knowledgeable about the menu.",
                "I had a wonderful experience at this restaurant. The food was flavorful and the presentation was beautiful.",
                "The restaurant had a great atmosphere. The lighting and music created a cozy and inviting environment.",
                "The food was absolutely amazing! I loved the flavors and the presentation was beautiful.",
                "I had a terrible experience at this restaurant. The service was slow and the food was cold.",
                "The ambiance was perfect for a romantic dinner. The lighting and music created a cozy atmosphere.",
                "I would highly recommend this place to anyone looking for a great dining experience. The staff was friendly and attentive, and the food was delicious.",
                "The portions were small and overpriced. I expected more for the price I paid.",
            ],
        }
    )


@pytest.fixture
def mixed_col():
    return pd.DataFrame({"strange col": [1, 2, 3, "a", "b", "c", None, 4.5, 5.6, "d", "e", None, 6.7, 7.8, "f", 8.9, 9.0, "g"]})


@pytest.fixture
def nan_col():
    return pd.DataFrame({"nan col": [None] * 18})


@pytest.fixture
def const_col():
    return pd.DataFrame({"const col": ["Pizza"] * 18})


@pytest.fixture
def one_sample_col():
    return pd.DataFrame({"one sample col": ["Pizza"] * 17 + ["Burger"]})


# 54 samples dfs
@pytest.fixture
def base_df_54(base_df):
    return pd.concat([base_df] * 3, ignore_index=True)


@pytest.fixture
def unbalanced_df(base_df_54):
    df = base_df_54.copy()
    df["pet"] = ["cat"] * 52 + ["dog"] * 2
    return df
