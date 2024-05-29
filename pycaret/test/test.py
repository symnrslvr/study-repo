import requests
import pytest
import json

url = "http://127.0.0.1:5000/"
def test_post_test():
    
    payload = {
        
            "Jsc": 10.71,
            "Voc": 0.19,
            "FF": 33.439999,
            "Efficiency": 0.66,
            "Sr": 9.70,
            "Sp": 19.370001,
            "Temp": 450
        }
    
    

    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    assert response.status_code == 200
    
    
if __name__ == "__main__":
    test_post_test()
