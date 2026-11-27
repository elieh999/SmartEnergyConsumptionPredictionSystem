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

## Models

Multiple Linear Regression is the machine learning baseline. It fits this project because the target is a continuous numerical value.

The baseline was trained on 23,780 chronological training observations and tested on 5,096 later observations. Its test MAE is 0.386 kW, its RMSE is 0.524 kW, and its R squared value is 0.456. These values will be compared with the LSTM results when that model is trained.

The deep learning model is Long Short Term Memory, LSTM. LSTM is suitable for this project because electricity consumption is time based data. It will use the completed readings from the previous 24 hours to predict the average global active power for the following hour. The planned network has one LSTM layer with 32 units and one output neuron.

The LSTM sequence preparation is complete. Each sample contains 24 completed hourly observations with 11 input features per hour and predicts the following hour. This produced 23,636 training sequences, 5,050 validation sequences, and 5,050 test sequences. Windows containing missing hours were excluded. The sequence arrays are generated locally and are not stored in GitHub because they can be recreated from the prepared data.

The LSTM was trained for 13 epochs with a batch size of 128 and a learning rate of 0.001. On the validation data, it achieved an MAE of 0.380 kW, an RMSE of 0.545 kW, and an R squared value of 0.626. The final test evaluation is completed separately so that the test data remains untouched during training.

On the untouched test data, the LSTM achieved an MAE of 0.329 kW, an RMSE of 0.476 kW, and an R squared value of 0.553. It produced 5,050 next hour predictions.

Both models were evaluated chronologically. For the final comparison, they were matched to the same 5,050 test hours. The LSTM had the lower MAE and RMSE and the higher R squared value.

## Result graphs

The project includes report ready graphs for the first week of the test period: actual electricity consumption compared with Multiple Linear Regression predictions, actual electricity consumption compared with LSTM predictions, and the LSTM training and validation loss. It also includes an MAE and RMSE comparison graph.

The comparison graph uses the 5,050 test hours shared by both models. The LSTM sequence preparation removes incomplete 24 hour windows, so this matching step makes the comparison fair.

## Files

* `prepare_energy_data.py` prepares the data for modeling.
* `train_linear_regression.py` trains and evaluates the Multiple Linear Regression baseline.
* `prepare_lstm_sequences.py` creates the 24 hour input sequences for the LSTM model.
* `train_lstm.py` trains the one layer 32 unit LSTM using the training and validation sequences.
* `reports/preprocessing_summary.json` contains the preprocessing results and row counts.
* `reports/lstm_sequence_summary.json` contains LSTM sequence counts and input shapes.
* `results/lstm_training_history.json` and `results/lstm_validation_metrics.json` contain the LSTM training outcome.
* `results/lstm_test_metrics.json` and `results/lstm_test_predictions.csv` contain the final LSTM test results.
* `create_result_graphs.py` creates report ready prediction, loss, and error comparison graphs.
* `energy_dashboard.py` opens an interactive desktop dashboard using the saved test predictions.
* `results/figures/` contains the generated graphs.
* `results/model_comparison_metrics.json` contains the metrics calculated on the same 5,050 test hours for both models.
* `results/` contains model metrics, coefficients, and test predictions.
* `models/linear_regression_model.json` contains the learned baseline model parameters.
* `docs/AI_Project_Report.docx` is the current project report.
* `data/README.md` explains how to get the original dataset.

The raw dataset and generated CSV files are not stored here because they are large. They can be recreated by downloading the original UCI data and running the preprocessing script.

## Run the preprocessing script

1. Download the dataset from the UCI link above.
2. Extract `household_power_consumption.txt` into `data/raw/`.
3. Install pandas with `pip install pandas`.
4. Run `python prepare_energy_data.py`.

Run `python prepare_energy_data.py --help` to see the available paths. For example, you can
keep the downloaded file elsewhere and write generated files to temporary folders:

```bash
python prepare_energy_data.py --input path/to/household_power_consumption.txt \
  --processed-dir output/data --reports-dir output/reports
```

## Create the result graphs

After the prediction files have been created, run:

```bash
python create_result_graphs.py
```

## Open the project dashboard

The dashboard does not retrain anything. It reads the saved test predictions and
lets you inspect a test hour, compare both model predictions, and view a small
chart around that hour.

```bash
python energy_dashboard.py
```

## Reproduce the technical work

Use Python 3.11 or later, install the packages, then run these commands in order:

```bash
pip install -r requirements.txt
python prepare_energy_data.py
python train_linear_regression.py
python prepare_lstm_sequences.py
python train_lstm.py --evaluate-test
python create_result_graphs.py
```

The raw UCI file must be in `data/raw/` before starting. The tests can be run with `python -m pytest -q`.

## AI acknowledgement

AI tools were used for understanding the project requirements, dataset exploration, and preprocessing code guidance. We reviewed the preprocessing process and remain responsible for the project decisions, analysis, and final report.
