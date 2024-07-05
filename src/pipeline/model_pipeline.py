import sys
import os
import pandas as pd
import numpy as np
from src.exceptions import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict_pipeline(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            feature_data_scaled = preprocessor.transform(features)
            preds = model.predict(feature_data_scaled)
            return preds
            
        except Exception as e:
            raise CustomException(e,sys)
        
class PredictData:
    def __init__(self, month:int, day:int):
        self.month = month
        self.day = day
    def get_data_as_data_frame(self):
        try:
            data_dict = {
                'month' : [self.month],
                'day' : [self.day]
            }
            return pd.DataFrame(data_dict)
        
        except Exception as e:
            raise CustomException(e,sys)