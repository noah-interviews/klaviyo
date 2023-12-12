import argparse
import json
import numpy as np
import os
import pandas as pd
import pickle
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import collect_and_prep_prophet_data


def main():
    # Parse args
    parser = argparse.ArgumentParser(description='Train and predict a time series model.')
    parser.add_argument('exp_name', type=str, help='Experiment name, all assets will be saved to a directory named this.')
    parser.add_argument('train_csv_path', type=str, help='Path to the training data CSV data file.')
    parser.add_argument('test_csv_path', type=str, help='Path to the test data CSV data file.')

    args = vars(parser.parse_args())

    # Collect and format data
    train_df = collect_and_prep_prophet_data(args['train_csv_path'], log_transform_series_col=True)
    # Don't impute series for test data
    test_df = collect_and_prep_prophet_data(args['test_csv_path'], log_transform_series_col=True)

    # Define Prophet model
    model = Prophet(growth='linear',
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    changepoint_range=1.0,
                    seasonality_mode='additive'
    )
    model.add_country_holidays(country_name='US')

    # Fit Prophet model
    model.fit(train_df)

    # Predict on training set to evaluate on training data
    train_forecast = model.predict(train_df)
    train_forecast['yhat'] = np.exp(train_forecast['yhat']).astype(int)
    train_forecast['yhat_lower'] = np.exp(train_forecast['yhat_lower']).astype(int)
    train_forecast['yhat_upper'] = np.exp(train_forecast['yhat_upper']).astype(int)
    eval_metrics = {}
    eval_metrics['train_mse'] = mean_squared_error(np.exp(train_df['y']).astype(int), train_forecast['yhat'])
    eval_metrics['train_mae'] = mean_absolute_error(np.exp(train_df['y']).astype(int), train_forecast['yhat'])
    print("Training MSE:", eval_metrics['train_mse'])
    print("Training MAE:", eval_metrics['train_mae'])

    # Make forecasts and transform/consolidate prediction data
    forecast = model.predict(test_df)
    forecast['yhat_lower'] = np.exp(forecast['yhat_lower']).astype(int)
    forecast['yhat_upper'] = np.exp(forecast['yhat_upper']).astype(int)
    forecast['yhat'] = np.exp(forecast['yhat']).astype(int)
    forecast['logy'] = test_df['y']
    forecast['y'] = np.exp(forecast['logy']).astype(int)
    
    # Evaluate on test data
    eval_metrics['test_mse'] = mean_squared_error(forecast['y'], forecast['yhat'])
    eval_metrics['test_mae'] = mean_absolute_error(forecast['y'], forecast['yhat'])
    print("Test MSE:", eval_metrics['test_mse'])
    print("Test MAE:", eval_metrics['test_mae'])

    non_zero_forecast = forecast.query("y > 0.0")
    eval_metrics['test_nonzero_mse'] = mean_squared_error(non_zero_forecast['y'], non_zero_forecast['yhat'])
    eval_metrics['test_nonzero_mae'] = mean_absolute_error(non_zero_forecast['y'], non_zero_forecast['yhat'])
    print("Test (non-zero) MSE:", eval_metrics['test_nonzero_mse'])
    print("Test (non-zero) MAE:", eval_metrics['test_nonzero_mae'])

    # Store model and other (meta)data
    os.makedirs(args['exp_name'], exist_ok=False)

    with open(f"{args['exp_name']}/args.json", 'w') as file:
        json.dump(args, file, indent=4)

    with open(f"{args['exp_name']}/eval_metrics.json", 'w') as file:
        json.dump(eval_metrics, file, indent=4)

    forecast.to_pickle(f"{args['exp_name']}/test_preds_actuals.csv")

    with open(f"{args['exp_name']}/model.pkl", 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    main()