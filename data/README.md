# Data

The original data is not included in this repository because it is large.

Download the Individual Household Electric Power Consumption dataset from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption

After extracting the download, place `household_power_consumption.txt` inside `data/raw/` and run `python prepare_energy_data.py`.

The script will create the prepared hourly data, train, validation, and test splits inside `data/processed/`.
