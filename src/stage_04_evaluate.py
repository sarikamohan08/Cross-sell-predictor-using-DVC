import pandas as pd
import argparse
from src.utils.common import (
    read_params,
    save_reports,
)
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score)
import logging
import joblib
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=logging_str)


def eval_metrics(actual, pred):
     
    accuracy=float(accuracy_score(pred,actual))
    precision = float(precision_score( pred,actual))
    recall = float(recall_score(pred,actual))
    f1=float(f1_score(pred,actual))
    return accuracy,precision,recall,f1

def evaluate(config_path):
    config = read_params(config_path)

    artifacts = config["artifacts"]
    test_data_path = artifacts["split_data"]["test_path"]
    model_path = artifacts["model_path"]
    target = config["base"]["target_col"]
    scores_file = artifacts["reports"]["scores"]

    test = pd.read_csv(test_data_path, sep=",")

    test_y = test[target]
    test_x = test.drop(target, axis=1)

    lr = joblib.load(model_path)
    logging.info(f"model is loaded from {model_path}")

    predicted_values = lr.predict(test_x)

    accuracy,precision,recall,f1  = eval_metrics(predicted_values,test_y)
    scores = {
    "accuracy_score": accuracy,
    "precision_score": precision,
    "recall_score": recall,
    "f1_score": f1
    }

    save_reports(scores_file, scores)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        data = evaluate(config_path=parsed_args.config)
        logging.info("evaluation stage completed")

    except Exception as e:
        logging.error(e)
        # raise e