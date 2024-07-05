import os
import sys
import dill
from dataclasses import dataclass

from src.exceptions import CustomException
from src.logger import logging
from src.utils import model_evaluate,save_object

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

@dataclass
class ModelTrainerConfig:
    model_trainer_obj_path = os.path.join('artifacts','model.pkl')

class ModelTraining:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def model_training(self, train_arr, test_arr):
        logging.info('model training initated')
        try:
            x_train, y_train, x_test, y_test = (
                train_arr[:,:2],
                train_arr[:,-1],
                test_arr[:,:2],
                test_arr[:,-1]
            )
            models = {
                'linear_regression' : LinearRegression(),
                'KNN' : KNeighborsRegressor(),
                'Decision_tree' : DecisionTreeRegressor(),
                'xgboost' : XGBRegressor()
            }

            model_report:dict = model_evaluate(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, models = models)
            best_model_score = min(model_report.values())
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            save_object(
                file_path = self.model_trainer_config.model_trainer_obj_path,
                obj = best_model
            )

            # return best_model_score
            predictions = best_model.predict(x_test)
            rmse = mean_squared_error(y_test, predictions)
            return rmse
        
        
        except Exception as e:
            raise CustomException(e,sys)
