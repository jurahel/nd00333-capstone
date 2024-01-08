# Source: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-existing-model
import json
import os
import pandas as pd
import numpy as np
import pickle
import joblib



def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    
    
    try:
        model = joblib.load(model_path)
        logger.info("Loaded successfully...")
    except Exception as e:
        error = str(e)
        return error
     
def run(data):
    try:      
        data = json.loads(data)['data']
        data = pd.DataFrame.from_dict(data)
        result = model.predict(data)
        
        return result.tolist()
        #return json.dumps({"result": result.tolist()})

    except Exception as e:
        error = str(e)
        return error

