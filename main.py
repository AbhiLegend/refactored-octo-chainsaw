import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import json
import pandas as pd
import requests
import zipfile
import io

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.pipeline.column_mapping import ColumnMapping

from datetime import datetime
from sklearn import datasets, ensemble

from evidently.dashboard import Dashboard
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard.tabs import DataDriftTab, NumTargetDriftTab, RegressionPerformanceTab
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles




"""## Bicycle Demand Data

This step automatically downloads the bike dataset from UCI. This version is slightly different from the dataset used in Kaggle competition. If you want the example to be identical to the one in the Evidently blog "How to break a model in 20 days", you can manually download the dataset from Kaggle: https://www.kaggle.com/c/bike-sharing-demand/data 

And add this code:

raw_data['mnth'] = raw_data.index.map(lambda x : x.month)

raw_data['hr'] = raw_data.index.map(lambda x : x.hour)

raw_data['weekday'] = raw_data.index.map(lambda x : x.weekday() + 1)
"""

#load data
content = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip").content
with zipfile.ZipFile(io.BytesIO(content)) as arc:
    raw_data = pd.read_csv(arc.open("day.csv"), header=0, sep=',', parse_dates=['dteday'], index_col='dteday')

raw_data.head()

"""## Regression Model

### Model training
"""
content = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip").content
with zipfile.ZipFile(io.BytesIO(content)) as arc:
    raw_data = pd.read_csv(arc.open("hour.csv"), header=0, sep=',', parse_dates=['dteday'], index_col='dteday')

raw_data.head()

"""## Regression Model

### Model training
"""

target = 'cnt'
prediction = 'prediction'
numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'weekday']
categorical_features = ['season', 'holiday', 'workingday']

reference = raw_data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00']
current = raw_data.loc['2011-01-29 00:00:00':'2011-02-28 23:00:00']

reference.head()

regressor = ensemble.RandomForestRegressor(random_state = 0, n_estimators = 50)

regressor.fit(reference[numerical_features + categorical_features], reference[target])

ref_prediction = regressor.predict(reference[numerical_features + categorical_features])
current_prediction = regressor.predict(current[numerical_features + categorical_features])

reference['prediction'] = ref_prediction
current['prediction'] = current_prediction

"""### Model Perfomance """

column_mapping = ColumnMapping()

column_mapping.target = target
column_mapping.prediction = prediction
column_mapping.numerical_features = numerical_features
column_mapping.categorical_features = categorical_features

regression_perfomance_dashboard = Dashboard(tabs=[RegressionPerformanceTab()])
regression_perfomance_dashboard.calculate(reference, None, column_mapping=column_mapping)

# regression_perfomance_dashboard.show()

regression_perfomance_dashboard.save("./static/index.html")

"""##  Week 1"""

regression_perfomance_dashboard.calculate(reference, current.loc['2011-01-29 00:00:00':'2011-02-07 23:00:00'], 
                                            column_mapping=column_mapping)

# regression_perfomance_dashboard.show()

regression_perfomance_dashboard.save("./static/regression_performance_after_week1.html")

target_drift_dashboard = Dashboard(tabs=[NumTargetDriftTab()])
target_drift_dashboard.calculate(reference, current.loc['2011-01-29 00:00:00':'2011-02-07 23:00:00'], 
                                   column_mapping=column_mapping)

# target_drift_dashboard.show()

target_drift_dashboard.save("./static/target_drift_after_week1.html")

"""## Week 2"""

regression_perfomance_dashboard.calculate(reference, current.loc['2011-02-07 00:00:00':'2011-02-14 23:00:00'], 
                                            column_mapping=column_mapping)

# regression_perfomance_dashboard.show()

regression_perfomance_dashboard.save("./static/regression_performance_after_week2.html")

target_drift_dashboard.calculate(reference, current.loc['2011-02-07 00:00:00':'2011-02-14 23:00:00'], 
                                   column_mapping=column_mapping)

# target_drift_dashboard.show()

target_drift_dashboard.save("./static/target_drift_after_week2.html")

"""## Week 3"""

regression_perfomance_dashboard.calculate(reference, current.loc['2011-02-15 00:00:00':'2011-02-21 23:00:00'], 
                                            column_mapping=column_mapping)

# regression_perfomance_dashboard.show()

regression_perfomance_dashboard.save("./static/regression_performance_after_week3.html")

target_drift_dashboard.calculate(reference, current.loc['2011-02-15 00:00:00':'2011-02-21 23:00:00'], 
                                   column_mapping=column_mapping)

# target_drift_dashboard.show()

target_drift_dashboard.save("./static/target_drift_after_week3.html")

"""## Data Drift"""

column_mapping = ColumnMapping()

column_mapping.numerical_features = numerical_features

data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
data_drift_dashboard.calculate(reference, current.loc['2011-01-29 00:00:00':'2011-02-07 23:00:00'], 
                                   column_mapping=column_mapping)

# data_drift_dashboard.show()

data_drift_dashboard.save("./static/data_drift_dashboard_after_week1.html")

"""## Data Drift Week 2"""
column_mapping = ColumnMapping()
column_mapping.numerical_features = numerical_features
data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
data_drift_dashboard.calculate(reference, current.loc['2011-02-07 00:00:00':'2011-02-14 23:00:00'],
                                   column_mapping=column_mapping)
data_drift_dashboard.save("./static/data_drift_dashboard_after_week2.html")


#set column mapping for Evidently Profile
data_columns = ColumnMapping()
data_columns.numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'weekday']
data_columns.categorical_features = ['season', 'holiday', 'workingday']
#evaluate data drift with Evidently Profile
def eval_drift(reference, production, column_mapping):
    data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    data_drift_profile.calculate(reference, production, column_mapping=column_mapping)
    report = data_drift_profile.json()
    json_report = json.loads(report)

    drifts = []

    for feature in column_mapping.numerical_features + column_mapping.categorical_features:
        drifts.append((feature, json_report['data_drift']['data']['metrics'][feature]['drift_score']))

    return drifts

#set reference dates
reference_dates = ('2011-01-01 00:00:00','2011-01-28 23:00:00')

#set experiment batches dates
experiment_batches = [
    ('2011-01-01 00:00:00','2011-01-29 23:00:00'),
    ('2011-01-29 00:00:00','2011-02-07 23:00:00'),
    ('2011-02-07 00:00:00','2011-02-14 23:00:00'),
    ('2011-02-15 00:00:00','2011-02-21 23:00:00'),  
]

#log into MLflow
client = MlflowClient()

#set experiment
mlflow.set_experiment('Data Drift Evaluation with Evidently')

#start new run
for date in experiment_batches:
    with mlflow.start_run() as run: #inside brackets run_name='test'
        
        # Log parameters
        mlflow.log_param("begin", date[0])
        mlflow.log_param("end", date[1])

        # Log metrics
        metrics = eval_drift(raw_data.loc[reference_dates[0]:reference_dates[1]], 
                             raw_data.loc[date[0]:date[1]], 
                             column_mapping=data_columns)
        for feature in metrics:
            mlflow.log_metric(feature[0], round(feature[1], 3))

        print(run.info)






app = FastAPI()

app.mount("/", StaticFiles(directory="static",html = True), name="static")
