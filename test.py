import requests

url = 'http://localhost:9696/predict'

# A fake patient profile
customer = {
    "age": 35,
    "sex": "male",
    "bmi": 27.5,
    "children": 2,
    "smoker": "yes",
    "region": "southwest"
}

# Send the data to our Flask app
response = requests.post(url, json=customer).json()

# Print the result
print(f"Predicted Medical Cost: ${response['predicted_medical_cost']:,.2f}")