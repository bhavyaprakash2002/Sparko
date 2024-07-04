import os
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.exceptions import CustomException
from ..logger import logging

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.xlsx')
    test_data_path:str=os.path.join('artifacts','test.xlsx')
    raw_data_path:str=os.path.join('artifacts','data.xlsx')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('data ingestion initiated')
        try:
            data = pd.read_excel(r'notebooks\big_data_1.xlsx')

            logging.info('dataset collected')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            data.to_excel(self.ingestion_config.raw_data_path)
            logging.info('train_test_split initiated')
            train_set, test_set = train_test_split(data, random_state=42)
            train_set.to_excel(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_excel(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info('data ingestion completed')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
