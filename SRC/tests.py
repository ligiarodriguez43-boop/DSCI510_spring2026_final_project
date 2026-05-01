import requests
import json

# Loads API Key
with open("api_key.txt", "r") as f:
    API_KEY = f.read().strip()

BASE    = "https://clinicaltrialsapi.cancer.gov/api/v2"
headers = {"X-API-KEY": API_KEY}

#Fetches Disease
print("Fetching colon cancer disease terms...")

response = requests.get(
    f"{BASE}/diseases",
    headers=headers,
    params={
        "name": "colon",
        "size": 300
    }
)
print(f"Status: {response.status_code}")
print(f"URL:    {response.url}")
response.raise_for_status()

raw = response.json()

diseases = (
    raw.get("terms") or
    raw.get("data")  or
    raw.get("results") or
    raw.get("diseases") or
    []
)
print(f"Records found:  {len(diseases)}")

#  Save to JSON File
if diseases:
    with open("colon_cancer_diseases.json", "w") as f:
        json.dump(diseases, f, indent=2)
    print("\nSaved to colon_cancer_diseases.json")
    print(json.dumps(diseases[0], indent=2))
