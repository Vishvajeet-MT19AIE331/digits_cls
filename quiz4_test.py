from api.app import app
import pytest


# define function to test the response of aap.py
def test_get_root():
    response = app.test_client().get("/predictor")
    assert response.status_code == 200
    assert (response.get_data() == b"<p>0</p>"+suffix or
     response.get_data() == b"<p>1</p>"+suffix or
     response.get_data() == b"<p>2</p>"+suffix or
     response.get_data() == b"<p>3</p>"+suffix or
     response.get_data() == b"<p>4</p>"+suffix or
     response.get_data() == b"<p>5</p>"+suffix or
     response.get_data() == b"<p>6</p>"+suffix or
     response.get_data() == b"<p>7</p>"+suffix or
     response.get_data() == b"<p>8</p>"+suffix or
     response.get_data() == b"<p>9</p>" )
    
# define function to test the (post) response of aap.py
def test_post_root():
    suffix = "post suffix"
    response = app.test_client().post("/predictor_post", json={"suffix":suffix})
    assert response.status_code == 200    
    assert (response.get_json()['op'] == b"<p>0</p>"+suffix or
     response.get_json()['op'] == b"<p>1</p>"+suffix or
     response.get_json()['op'] == b"<p>2</p>"+suffix or
     response.get_json()['op'] == b"<p>3</p>"+suffix or
     response.get_json()['op'] == b"<p>4</p>"+suffix or
     response.get_json()['op'] == b"<p>5</p>"+suffix or
     response.get_json()['op'] == b"<p>6</p>"+suffix or
     response.get_json()['op'] == b"<p>7</p>"+suffix or
     response.get_json()['op'] == b"<p>8</p>"+suffix or
     response.get_json()['op'] == b"<p>9</p>" )
