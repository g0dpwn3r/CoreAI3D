import requests

# Assuming the server is running on localhost:8080
response = requests.post("http://localhost:8080/api/v1/generate_key")
if response.status_code == 200:
    data = response.json()
    # The API key is nested in data["data"]["api_key"]
    api_key = data.get("data", {}).get("data", {}).get("api_key")
    if api_key:
        print(f"New API Key: {api_key}")
    else:
        print("API key not found in response.")
        print(f"Response content: {response.text}")
else:
    print(f"Failed to generate API key. Status code: {response.status_code}")
    print(f"Response content: {response.text}")