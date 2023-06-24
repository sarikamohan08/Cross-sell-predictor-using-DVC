import pandas as pd
import argparse
from src.utils.common import (
    read_params,
    create_dir,
    save_reports,
)
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=logging_str)



def train(config_path):
    config = read_params(config_path)
    artifacts = config["artifacts"]

    split_data = artifacts["split_data"]
    processed_data_dir = split_data["processed_data_dir"]
    test_data_path = split_data["test_path"]
    train_data_path = split_data["train_path"]

    base = config["base"]
    random_seed = base["random_state"]
    target = base["target_col"]

    reports = artifacts["reports"]
    reports_dir = reports["reports_dir"]
    params_file = reports["params"]

    RandomForestClassifier_params = config["estimators"]["RandomForestClassifier"]["params"]
    n_estimators = RandomForestClassifier_params["n_estimators"]

    train = pd.read_csv(train_data_path , sep=',')
    #print(train.head(2))
    print(train.columns)
    train_x = train.drop(target, axis=1)
    train_y = train[target]
    print(train_y)
   
    #print(train_y.head(2))
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_seed)
    rf.fit(train_x, train_y)

    model_dir = artifacts["model_dir"]
    model_path = artifacts["model_path"]

    create_dir([model_dir, reports_dir])

    params = {
        "n_estimators": n_estimators,
       
    }

    save_reports(params_file, params)

    joblib.dump(rf, model_path)

    logging.info(f"model saved at {model_path}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        data = train(config_path=parsed_args.config)
        logging.info("training stage completed")

    except Exception as e:
        logging.error(e)
        # raise e