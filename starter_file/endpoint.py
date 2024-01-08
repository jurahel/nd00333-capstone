

import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://0d469fee-33c0-4ce9-9dac-2767356110c9.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = 'zuv0yY9prOPHFssuYSFWDJPhnVMxgJqG'

# Two sets of data to score, so we get two results back
data = {"data":
      [{"Pregnancies": 6, 
     "Glucose": 148, 
     "BloodPressure": 72, 
     "SkinThickness": 35, 
     "Insulin": 0, 
     "BMI": 33.5, 
     "DiabetesPedigreeFunction": 0.627, 
     "Age": 50},

    {"Pregnancies": 1, 
     "Glucose": 85, 
     "BloodPressure": 66, 
     "SkinThickness": 29, 
     "Insulin": 20, 
     "BMI": 26.5, 
     "DiabetesPedigreeFunction": 0.351, 
     "Age": 31},
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
#headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
