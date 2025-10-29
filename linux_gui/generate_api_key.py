import requests

# Assuming the server is running on localhost:8080
response = requests.post("http://localhost:8080/api/v1/generate_key")
if response.status_code == 200:
    data = response.json()
    api_key = data.get("api_key")
    print(f"New API Key: {api_key}")
else:
    print(f"Failed to generate API key. Status code: {response.status_code}")