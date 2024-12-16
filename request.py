import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Age':20, 'Tonnage':48, 'passengers':6, 'length':6, 'cabins':6, 'passenger_density':32})

print(r.json())