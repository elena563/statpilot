import pandas as pd
import pytest

from modules.analysis import analyze_num, analyze_qual, analyze_text, is_text, read_csv_sep


def test_is_text():
    # categorical data
    cat_series = pd.Series(["Pizza", "Ananas", "Sushi", "Pizza", "Pasta", "Salad", "Pizza Margherita", "Burger", "Tacos", "Mexican Burrito", "Pizza", "Ramen"])
    assert not is_text(cat_series)

    # numeric data
    numeric_series = pd.Series([1, 2, 3, 4, 5])
    assert not is_text(numeric_series)

    # boolean data
    bool_series = pd.Series([True, False, True, False])
    assert not is_text(bool_series)

    # empty series
    empty_series = pd.Series([])
    assert not is_text(empty_series)

    # free text data
    free_text_series = pd.Series(
        [
            "The food was absolutely amazing! I loved the flavors and the presentation was beautiful.",
            "I had a terrible experience at this restaurant. The service was slow and the food was cold.",
            "The ambiance was perfect for a romantic dinner. The lighting and music created a cozy atmosphere.",
            "I would highly recommend this place to anyone looking for a great dining experience. The staff was friendly and attentive, and the food was delicious.",
            "The portions were small and overpriced. I expected more for the price I paid.",
            "Horrible",
            "The food was cold and tasteless. I will not be coming back.",
        ]
    )
    assert is_text(free_text_series)


def test_analyze_num(base_df, tmp_path):
    stats, plots = analyze_num(base_df, ["age", "score"], tmp_path)
    assert "age" in stats
    assert "score" in stats
    assert len(plots) > 0


def test_analyze_qual(base_df, tmp_path):
    stats, plots = analyze_qual(base_df, ["city", "category"], tmp_path)
    assert "city" in stats
    assert "category" in stats
    assert len(plots) > 0


def test_analyze_text(base_df, tmp_path):
    stats, plots = analyze_text(base_df, ["review"], tmp_path)
    assert "review" in stats
    assert len(plots) > 0


@pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
def test_read_csv_sep(tmp_path, delimiter):
    csv = tmp_path / "test.csv"
    csv.write_text(f"a{delimiter}b{delimiter}c\n1{delimiter}2{delimiter}3\n4{delimiter}5{delimiter}6")
    with open(csv) as f:
        df = read_csv_sep(f)
    assert df.shape == (2, 3)
