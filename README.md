# Smart Energy Consumption Prediction System

This project is about predicting household electricity consumption one hour in advance. We are using historical electricity readings to predict the future value of global active power.

It is a supervised machine learning regression project because the output is a number, not a category.

## Team

* Elie Helou 2023 13 104
* Kamal Jammoul 2024 10 293

## Dataset

We are using the Individual Household Electric Power Consumption dataset from the UCI Machine Learning Repository.

The original data contains 2,075,259 electricity measurements collected every minute from December 2006 to November 2010 in one household. It includes date, time, global active power, global reactive power, voltage, global intensity, and three sub metering values.

Dataset link: https://archive.ics.uci.edu/dataset/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption

## What has been done so far

The data was cleaned by converting question marks into missing values and removing incomplete records. Date and time were combined into one date time variable, and the electrical values were converted to numbers.

The minute records were grouped into hourly observations. Mean values were used for global active power, global reactive power, voltage, and global intensity. The three sub metering values were summed.

We created hour, day of week, month, weekend status, and historical power features. Power lag 1 is the electricity consumption one hour earlier. Power lag 2 is two hours earlier. Power lag 24 is the consumption at the same time on the previous day.

The target is global active power one hour in the future. The final prepared data contains 33,972 hourly observations. The data was divided chronologically into 70% training, 15% validation, and 15% test data. The input features were normalized using the training data mean and standard deviation.

## Model

The selected machine learning model is Multiple Linear Regression. It fits this project because the target is a continuous numerical value. It will be used as the first model and baseline before trying more advanced models later.

## Files

* `prepare_energy_data.py` prepares the data for modeling.
* `reports/preprocessing_summary.json` contains the preprocessing results and row counts.
* `docs/AI_Project_Report.docx` is the current project report.
* `data/README.md` explains how to get the original dataset.

The raw dataset and generated CSV files are not stored here because they are large. They can be recreated by downloading the original UCI data and running the preprocessing script.

## Run the preprocessing script

1. Download the dataset from the UCI link above.
2. Extract `household_power_consumption.txt` into `data/raw/`.
3. Install pandas with `pip install pandas`.
4. Run `python prepare_energy_data.py`.

## AI acknowledgement

AI tools were used for understanding the project requirements, dataset exploration, and preprocessing code guidance. We reviewed the preprocessing process and remain responsible for the project decisions, analysis, and final report.
