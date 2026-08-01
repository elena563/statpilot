from io import BytesIO
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