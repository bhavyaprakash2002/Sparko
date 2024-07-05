from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.pipeline.model_pipeline import PredictData
from src.pipeline.model_pipeline import PredictPipeline

application = Flask(__name__)
app = application

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
        return render_template('home.html', results = result[0])
    
if __name__=='__main__':
    app.run(host="0.0.0.0", debug=True)
