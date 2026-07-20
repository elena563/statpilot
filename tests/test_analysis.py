import pandas as pd

from modules.analysis import is_text

def test_is_text():
    # categorical data
    cat_series = pd.Series(['Pizza', 'Ananas', 'Sushi', 'Pizza', 'Pasta', 'Salad', 'Pizza Margherita', 'Burger', 'Tacos', 'Mexican Burrito', 'Pizza', 'Ramen'])
    assert is_text(cat_series) == False

    # numeric data
    numeric_series = pd.Series([1, 2, 3, 4, 5])
    assert is_text(numeric_series) == False

    # boolean data
    bool_series = pd.Series([True, False, True, False])
    assert is_text(bool_series) == False

    # empty series
    empty_series = pd.Series([])
    assert is_text(empty_series) == False

    # free text data
    free_text_series = pd.Series(['The food was absolutely amazing! I loved the flavors and the presentation was beautiful.',
                                  'I had a terrible experience at this restaurant. The service was slow and the food was cold.',
                                  'The ambiance was perfect for a romantic dinner. The lighting and music created a cozy atmosphere.',
                                  'I would highly recommend this place to anyone looking for a great dining experience. The staff was friendly and attentive, and the food was delicious.',
                                  'The portions were small and overpriced. I expected more for the price I paid.',
                                  'Horrible', 'The food was cold and tasteless. I will not be coming back.',])
    assert is_text(free_text_series) == True