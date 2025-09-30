from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, r2_score
import os
import pandas as pd
import numpy as np

def calc_classification_metrics(y_train, y_val, y_test, train_pred, val_pred, test_pred):
    acc_train = accuracy_score(y_train, train_pred)
    acc_val = accuracy_score(y_val, val_pred)
    acc_test = accuracy_score(y_test, test_pred)

    res= {
        'train_accuracy': acc_train,
        'val_accuracy': acc_val,
        'test_accuracy': acc_test,
    }
    return res


def calc_regression_metrics(y_train, y_val, y_test, train_pred, val_pred, test_pred):
    mse_train = mean_squared_error(y_train, train_pred)
    mse_val = mean_squared_error(y_val, val_pred)
    mse_test = mean_squared_error(y_test, test_pred)
    mae_train = mean_absolute_error(y_train, train_pred)
    mae_val = mean_absolute_error(y_val, val_pred)
    mae_test = mean_absolute_error(y_test, test_pred)
    r2_train = r2_score(y_train, train_pred)
    r2_val = r2_score(y_val, val_pred)
    r2_test = r2_score(y_test, test_pred)

    res= {
        'train_mse': mse_train,
        'val_mse': mse_val,
        'test_mse': mse_test,
        'train_mae': mae_train,
        'val_mae': mae_val,
        'test_mae': mae_test,
        'train_r2': r2_train,
        'val_r2': r2_val,
        'test_r2': r2_test
    }
    return res

def save_metrics(persist, args):
    df = pd.DataFrame([persist])
    results_dir = f"{args.destination_dir}/{args.destination_file}"
    # check if the file exists
    if os.path.exists(results_dir):
        results_df = pd.read_csv(results_dir)
        results_df = pd.concat([results_df, df], axis = 0)
        results_df.to_csv(results_dir, index = False)
    else:
        df.to_csv(results_dir, index = False)

