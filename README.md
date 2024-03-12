# ProgrammingTask

## Repository Overview
This repository contains several Python scripts and Jupyter notebooks for the ProgrammingTask

Here's a brief overview of each file:

- `datahandler.py`: Contains methods for reading and preprocessing the data.
- `dataprepper.py`: Contains methods for preparing the targets and features for the models.
- `models.py`: Defines the forecasting model.
- `trainer.py`: Contains the method for training the model.
- `evaluator.py`: Contains the method for evaluating the model's performance.

The trained models are located in the "models" directory on the format:

`<dataset_name>/<model_type>/<forecast_steps>/<participantID>.pth`

The scalers are located in the "scalers" directory.

## Reports
The main report is located in the Jupyter Notebook:
- `Report.ipynb`

Detailed results of each forecasting step are provided in three separate Jupyter notebooks:

- `Report_forecast_30min.ipynb`
- `Report_forecast_60min.ipynb`
- `Report_forecast_90min.ipynb`

To avoid retraining all the models when running the notebooks, follow these steps:

1. Run all sections before the "Experiments" section.
2. Run all sections after the "Experiments" section.

## Installation
To install all the necessary libraries, run the following command in your terminal:

```bash
pip install -r requirements.txt

## Datamodel
https://illustrious-star-52d.notion.site/Datamodel-b12271138b9543c0938635e364278099?pvs=4
