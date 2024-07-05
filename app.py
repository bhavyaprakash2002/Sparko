from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.pipeline.model_pipeline import PredictData
from src.pipeline.model_pipeline import PredictPipeline
from xgboost import XGBRegressor


# consumer part begins
import boto3
import sagemaker
import os
from datetime import datetime
import csv
from io import StringIO
# consumer part ends


application = Flask(__name__)
app = application

s3 = boto3.client('s3')
sessions = sagemaker.Session()
region = sessions.boto_session.region_name
bucket = 'consumerusage'
csv_file_name = 'consumer_usage.csv'

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict_data():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = PredictData(month = request.form.get('month'),
                           day = request.form.get('day'))
        
        pred_df = data.get_data_as_data_frame()
        pred_pipeline = PredictPipeline()
        result = pred_pipeline.predict_pipeline(pred_df)

        csv_obj = s3.get_object(Bucket=bucket, Key=csv_file_name)
        csv_string = csv_obj['Body'].read().decode('utf-8')
        # Read the CSV content into a pandas DataFrame
        df = pd.read_csv(StringIO(csv_string))
        x = df.iloc[:,:2]
        y = df['usage']
        model = XGBRegressor()
        model.fit(x,y)
        
        dict_data = {
            'month' : [int(request.form.get('month'))],
            'day' : [int(request.form.get('month'))]
        }
        
        df_data = pd.DataFrame(dict_data)
        usage_preds = model.predict(df_data)
        

        return render_template('home.html', results = [result[0],usage_preds[0]])
    


                
if __name__=='__main__':
    app.run(host="0.0.0.0", debug=True)
