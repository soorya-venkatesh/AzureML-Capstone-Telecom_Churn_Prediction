import pickle
import json
import pandas as pd
import joblib
from azureml.core.model import Model


def init():
    global model
    try: 
        
        model_path = Model.get_model_path('rf_hyperdrive')
        model = joblib.load(model_path)
    except Exception as err:
        print('init method error: ' + str(err))

def run(data):
    try:
        data = pd.DataFrame(json.loads(data)['data'])
        
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error