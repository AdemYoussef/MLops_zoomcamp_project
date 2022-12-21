from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import CronSchedule
from score import ride_duration_prediction

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