import sqlite3
import kagglehub
import numpy as np
import pandas as pd
import apps.reporter as rpt
from pandas import DataFrame
from kagglehub import KaggleDatasetAdapter
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)


def download_and_extract_from_kagglehub(ds_path: str,
                                        ds_file_name: str,
                                        db_file_name: str,
                                        update_db: bool = False) -> DataFrame | None:
    """
    Download and extract data from kagglehub
    :param ds_path: path to kaggle dataset
    :param ds_file_name: name of kaggle dataset
    :param db_file_name: name of a database file
    :param update_db: update the database
    :return: DataFrame or None
    """
    ds_name = ds_path.split("/")[-1].replace('-', '_')

    # Note: to handle error: "SSL: CERTIFICATE_VERIFY_FAILED" or no connection to the server
    try:
        # for testing
        # raise Exception

        ds_data = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            ds_path,
            ds_file_name,
        )

        # Use only one time to initialize/update data (at first time)
        if not ds_data.empty:
            conn = sqlite3.connect(db_file_name)
            try:
                exist_conf = "replace" if update_db else "fail"
                ds_data.to_sql(ds_name, conn, if_exists=exist_conf, index=False)
            except ValueError:
                pass
            finally:
                conn.close()
    except Exception:
        conn = sqlite3.connect(db_file_name)
        ds_data = pd.read_sql(f"SELECT * FROM {ds_name}", conn)
        conn.close()

    return ds_data


def calc_class_metrics(y_test, y_pred):
    rp = rpt.Reporter()
    rp.tolerance = 4

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    rp.add_item("Confusion Matrix", rp.format_matrix(cm))
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    rp.add_item("Accuracy\n(Точність)", rp.format_value(accuracy))
    # Precision
    precision = precision_score(y_test, y_pred)
    rp.add_item("Precision\n(Влучність)", rp.format_value(precision))
    # Recall
    recall = recall_score(y_test, y_pred)
    rp.add_item("Recall\n(Повнота)", rp.format_value(recall))
    # F1-score
    f1 = f1_score(y_test, y_pred)
    rp.add_item("F1-score", rp.format_value(f1))

    # Print results
    rp.print_pd_report(f"Метрики")


def calc_regres_metrics(y_test, y_pred):
    rp = rpt.Reporter()
    rp.tolerance = 4

    # Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)
    rp.add_item("MAE", rp.format_value(mae))
    # Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    rp.add_item("MSE", rp.format_value(mse))
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    rp.add_item("RMSE", rp.format_value(rmse))
    # R2 - coefficient of determination
    r2 = r2_score(y_test, y_pred)
    rp.add_item("R²\n(коефіцієнт детермінації)", rp.format_value(r2))

    # Print results
    rp.print_pd_report(f"Метрики")
