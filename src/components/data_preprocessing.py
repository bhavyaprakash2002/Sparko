import os
import sys
import dill
from dataclasses import dataclass

import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataPreprocessingConfig:
    preprocessor_obj_path = os.path.join('artifacts','preprocessor.pkl')

class DataPreprocessing:
    def __init__(self):
        self.preprocessor_config = DataPreprocessingConfig()

    def preprocessing_steps(self):
        logging.info('preprocessing started')
        try:
            features = ['month','day']
            feature_pipeline = Pipeline([('scaler',StandardScaler(with_mean=False))])
            transformer = ColumnTransformer([('preprocessing', feature_pipeline, features)])

            logging.info('transformation steps defined')
            return transformer

        except Exception as e:
            raise CustomException
        
    def data_preprocessing(self,train_data_path, test_data_path):
        try:
            train_data = pd.read_excel(train_data_path)
            test_data = pd.read_excel(test_data_path)
            logging.info('data preprocessing initiated')
            transformation_obj = self.preprocessing_steps()

            target = ['irradiance']
            features = ['month','day']

            # train_data_input_features = train_data.drop(['year','irradiance'],axis=1)
            # test_data_input_features = test_data.drop(['year','irradiance'],axis=1)

            train_data_input_features = train_data.iloc[:,1:3]
            test_data_input_features = test_data.iloc[:,1:3]

            train_data_target = train_data['irradiance']
            test_data_target = test_data['irradiance']

            train_data_scaled = transformation_obj.fit_transform(train_data_input_features)
            test_data_scaled = transformation_obj.transform(test_data_input_features)

            train_arr = np.c_[train_data_scaled, np.array(train_data_target)]
            test_arr = np.c_[test_data_scaled, np.array(test_data_target)]
            logging.info("data prepocessing completed")

            save_object(
                file_path = self.preprocessor_config.preprocessor_obj_path,
                obj = transformation_obj
            
            )
            return(
                train_arr,
                test_arr,
                self.preprocessor_config.preprocessor_obj_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

