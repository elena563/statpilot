from io import BytesIO
from pathlib import Path
import re
import pytest


def test_no_file(client):
    response = client.post("/analyze", data={})
    assert b"No file submitted" in response.data


def test_wrong_extension(client):
    data = {"dataset": (BytesIO(b"col1,col2\n1,2"), "test.txt")}
    response = client.post("/analyze", content_type="multipart/form-data", data=data)
    assert b"must be a CSV" in response.data


def test_valid_csv(client):
    csv_content = b"age,score\n25,80.5\n30,90.0"
    data = {"dataset": (BytesIO(csv_content), "test.csv")}
    response = client.post("/analyze", content_type="multipart/form-data", data=data)
    assert b"error" not in response.data.lower()


@pytest.mark.parametrize("route", ["/analyze", "/model", "/", "/explain"])
def test_get_request(client, route):
    response = client.get(route)
    assert response.status_code == 200


def test_e2e_modeling(client):
    # Upload CSV
    csv_content = (Path(__file__).resolve().parent.parent / "static" / "test.csv").read_bytes()
    data = {"form_type": "train_form", "dataset": (BytesIO(csv_content), "test.csv")}
    response = client.post("/model", content_type="multipart/form-data", data=data)
    assert b"error" not in response.data.lower()

    m = re.search(r'name="session_id" value="([^"]+)"', response.get_data(as_text=True))
    sid = m.group(1)

    # Train model
    response = client.post("/model", data={"form_type": "train_form", "target": "age", "model": "Linear Regression", "session_id": sid})
    assert b"error" not in response.data.lower()

    # Test model
    input_data = {"score": "85.0", "city": "Milano", "category": "A", "review": "Good"}
    response = client.post("/model", data={"form_type": "test_form", "session_id": sid, **input_data})
    assert b"error" not in response.data.lower()
    assert b"age:" in response.data.lower()


def test_explain_no_files(client):
    response = client.post("/explain", data={"form_type": "global_form"})
    assert b"No file submitted" in response.data


def test_explain_wrong_model_extension(client):
    data = {"form_type": "global_form", "xtest": (BytesIO(b"num\n1"), "test.csv"), "model": (BytesIO(b"x"), "model.txt")}
    response = client.post("/explain", content_type="multipart/form-data", data=data)
    assert b"must be an ONNX model" in response.data


def test_explain_local_invalid_session(client):
    with pytest.raises(ValueError):
        client.post("/explain", data={"form_type": "local_form", "session_id": "not-a-uuid", "obs": "0"})


def test_explain_local_obs_errors(client):
    static_dir = Path(__file__).resolve().parent.parent / "static"
    csv_content = (static_dir / "xtest.csv").read_bytes()
    onnx_content = (static_dir / "model.onnx").read_bytes()

    response = client.post(
        "/explain", content_type="multipart/form-data", data={"form_type": "global_form", "xtest": (BytesIO(csv_content), "xtest.csv"), "model": (BytesIO(onnx_content), "model.onnx")}
    )

    sid = re.search(r'name="session_id" value="([^"]+)"', response.get_data(as_text=True)).group(1)

    res = client.post("/explain", data={"form_type": "local_form", "session_id": sid, "obs": "testo"})
    assert b"Observation index must be an integer" in res.data

    res = client.post("/explain", data={"form_type": "local_form", "session_id": sid, "obs": "99999"})
    assert b"Observation index is out of bounds" in res.data

    res = client.post("/explain", data={"form_type": "local_form", "session_id": sid, "obs": "-1"})
    assert b"Observation index is out of bounds" in res.data


def test_e2e_explain(client):
    static_dir = Path(__file__).resolve().parent.parent / "static"
    csv_content = (static_dir / "xtest.csv").read_bytes()
    onnx_content = (static_dir / "model.onnx").read_bytes()

    response = client.post(
        "/explain", content_type="multipart/form-data", data={"form_type": "global_form", "xtest": (BytesIO(csv_content), "xtest.csv"), "model": (BytesIO(onnx_content), "model.onnx")}
    )
    print(response.get_data(as_text=True)[:1000])

    sid = re.search(r'name="session_id" value="([^"]+)"', response.get_data(as_text=True)).group(1)

    res = client.post("/explain", data={"form_type": "local_form", "session_id": sid, "obs": "0"})
    assert b"error" not in response.data.lower()
    assert b"error" not in res.data.lower()
