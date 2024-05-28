import requests
import json

url = "http://127.0.0.1:5000/predict"
data = [
    {
        "Jsc": 10.71,
        "Voc": 0.19,
        "FF": 33.439999,
        "Efficiency": 0.66,
        "Sr": 9.70,
        "Sp": 19.370001,
        "Temp": 450
    },
    {
        "Jsc": 11.40,
        "Voc": 0.25,
        "FF": 38.529999,
        "Efficiency": 1.09,
        "Sr": 8.92,
        "Sp": 45.230000,
        "Temp": 450
    }
]

headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json.dumps(data), headers=headers)

print(response.json())
