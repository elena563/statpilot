from io import BytesIO
import re
import pytest

def test_no_file(client):
    response = client.post('/analyze', data={})
    assert b'No file submitted' in response.data

def test_wrong_extension(client):
    data = {'dataset': (BytesIO(b"col1,col2\n1,2"), 'test.txt')}
    response = client.post('/analyze', content_type='multipart/form-data', data=data)
    assert b'must be a CSV' in response.data

def test_valid_csv(client):
    csv_content = b"age,score\n25,80.5\n30,90.0"
    data = {'dataset': (BytesIO(csv_content), 'test.csv')}
    response = client.post('/analyze', content_type='multipart/form-data', data=data)
    assert b'error' not in response.data.lower()

@pytest.mark.parametrize("route", ["/analyze", "/model", "/", "/explain"])
def test_get_request(client, route):
    response = client.get(route)
    assert response.status_code == 200

from pathlib import Path


def test_e2e_modeling(client):
    # Upload CSV
    csv_content = (Path(__file__).resolve().parent.parent / 'static' / 'test.csv').read_bytes()
    data = {'form_type': 'train_form', 'dataset': (BytesIO(csv_content), 'test.csv')}
    response = client.post('/model', content_type='multipart/form-data', data=data)
    assert b'error' not in response.data.lower()

    m = re.search(r'name="session_id" value="([^"]+)"', response.get_data(as_text=True))
    sid = m.group(1)

    # Train model
    response = client.post('/model', data={'form_type': 'train_form', 'target': 'age', 'model': 'Linear Regression', 'session_id': sid})
    assert b'error' not in response.data.lower()

    # Test model
    input_data = {'score': '85.0', 'city': 'Milano', 'category': 'A', 'review': 'Good'}
    response = client.post('/model', data={'form_type': 'test_form', 'session_id': sid, **input_data})
    assert b'error' not in response.data.lower()
    assert b'age:' in response.data.lower()
