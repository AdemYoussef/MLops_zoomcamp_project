# MLops_zoomcamp_project
This is a follow-along practice MLops project with the goal of getting started and familiarized with The basics of `Machine Learning Operations`

This project was presented and delivered by [DataTalksClub](https://github.com/DataTalksClub/mlops-zoomcamp) (make sure to join Slack for the live Camps and the upcoming events / tutos)

Link to the full Youtube list [Here](https://www.youtube.com/playlist?list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK)

## READ THIS NOTE FIRST

Some of the tutos in this project are made with AWS applications, and I highly recommend getting an AWS account with around 30 to 40$ subscription to be able to re-do **Exactly** what was taught in this course, but you can for sure use other cloud providers such as Microsft Azure or GCP.
For me I couldn't get access to AWS resources, so I went for Azure, that's why i wasn't able to complete the course fully :pensive:

## About the Business problem

We have data records in `parquet file format` offered by the **NYC Taxi & Limousine Commission** which contains the **TLC Trip Record** which is essentially the data of a taxi ride from a point **A** to a point **B** in Newyork city with 17 features and the ***goal is to predict the duration of the ride requested by a new client***.
- Data can be found [Here](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

- Model used:
    An Xgboost linear regression model (details about the params in the `model_training.py` )


## Project Modules 

- Experimental Tracking / Model Artifact versioning & Tracking with `MLflow`
- Model Deployment with `Flask` (basic Flask application)
- Workflow orchestration with `Prefect2` 
- Model Monitoring with `Evidently` (Model performance & report generation)
- Near-live Data drift Detection & Vizualisation with `Evidently, Prometheus, Grafana & MongoDB`
- Testing / Test integration (`COMING SOON...`)
- Code quality (`COMING SOON...`)

## Tools & Environement

- Python==3.9
- Docker & Docker-compose
- IDE== VS code (VS code Remote ssh Extention)
- Azure VM (Memory optimized family, trust me, your PC will thank you for this)

## Project personal outcome/ Feedback

For me, this was really worth the time and effort because I'm in my early career as a Data Scientist and I highly recommend putting in, the time and effort in doing it to grow your skills.
And even, if you are already familiar with `**MLOps**`, you can still check out some parts of this project as it really **dives deep** into some **advanced concepts** with very well explanations (especially the **best practice part**: code quality, linters, AWS (Kinesis, Lambda, S3....), Terraform ...)

If you have any questions, you can freely message me on [Linkedin](https://www.linkedin.com/in/adem-youssef-277019176/)
