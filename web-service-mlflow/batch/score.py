#!/usr/bin/env python
# coding: utf-8

import pickle
import uuid
import sys
import pandas as pd
from sklearn.pipeline import make_pipeline
import mlflow

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from datetime import datetime
from dateutil.relativedelta import relativedelta

from prefect import task,flow,get_run_logger
from prefect.context import get_run_context

from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import CronSchedule





def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df['ride_id'] = generate_uuids(len(df))

    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts



def load_model(run_id):
    #logged_model = f"mlflow-artifacts:/1/{RUN_ID}/artifacts/model"
    logged_model = f'runs:/{run_id}/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model

def save_results(df, y_pred, run_id, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id

    df_result.to_parquet(output_file, index=False)

@task
def apply_model(input_file, run_id, output_file):

    logger = get_run_logger()
    logger.info(f'reading the data from {input_file}')
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    logger.info(f'loading the model with the RUN_ID = {run_id}')
    model = load_model(run_id)
    logger.info(f'model runing...')
    y_pred = model.predict(dicts)
    
    save_results(df, y_pred, run_id, output_file)
    return output_file

def get_paths(run_date, taxi_type):
    prev_month = run_date - relativedelta(months=1)
    year = prev_month.year
    month = prev_month.month 

    input_file = f'/home/adem/MLops_zoomcamp_project/data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'/home/adem/MLops_zoomcamp_project/web-service-mlflow/batch/output/{taxi_type}/{year:04d}-{month:02d}.parquet'

    return input_file, output_file

@flow
def ride_duration_prediction(
        taxi_type: str,
        run_id: str,
        run_date: datetime = None):

    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time
    
    
    input_file, output_file = get_paths(run_date, taxi_type)

    

    apply_model(
        input_file=input_file, 
        run_id=run_id, 
        output_file=output_file
        )


def run():

    taxi_type = sys.argv[1] # 'green'
    year = int(sys.argv[2]) # 2021
    month = int(sys.argv[3]) # 3
    RUN_ID = sys.argv[4] # '3742b6094a8f40558aab321accd39995'

    #RUN_ID = "3742b6094a8f40558aab321accd39995"
    MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("green-taxi-duration")
    #logged_model = f"mlflow-artifacts:/1/{RUN_ID}/artifacts/model"
    logged_model = f'runs:/{RUN_ID}/model'
    model = mlflow.pyfunc.load_model(logged_model)


    ride_duration_prediction(
        taxi_type=taxi_type,
        run_id=RUN_ID,
        run_date=datetime(year=year, month=month, day=1)
    )

    deployment = Deployment.build_from_flow(
    flow=ride_duration_prediction,
    name="ride_duration_prediction",
    parameters={
        "taxi_type": "green",
        "run_id": "3742b6094a8f40558aab321accd39995",
    },
    schedule=CronSchedule(cron="0 3 2 * *"), #at 3AM on the day of month 2, visit contab guru to test
    work_queue_name="batch_ride_duration_prediction",
    )
    deployment.apply()

    

    

if __name__ == '__main__':
    run()
    #python score.py green 2021 4 3742b6094a8f40558aab321accd39995






