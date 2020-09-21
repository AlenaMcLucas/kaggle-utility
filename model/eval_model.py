
# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s:   %(asctime)s\n%(message)s')
file_handler = logging.FileHandler('logs/eval.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# import libraries
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score, f1_score, log_loss, precision_score, recall_score, confusion_matrix
import pandas as pd
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as plt



# returns speceficity, support, false positive rate, and false negative rate
def classification_extra(y_true, y_pred):
    TP, TN, FP, FN = 0, 0, 0, 0

    for t,p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            TP += 1
        elif t == 1 and p == 0:
            FN += 1
        elif t == 0 and p == 1:
            FP += 1
        elif t == 0 and p == 0:
            TN += 1
        else:
            raise ValueError("There is a value that is not 0 or 1 -> t: {}, p: {}".format(t,p))

    return {"Speceficity": TN / (TN + FP), "Support": "1: {}, 0: {}".format(TP + FN, TN + FP),
            "False Positive Rate": FP / (FP + TN), # aka Type I Error
            "False Negative Rate": FN / (FN + TP)} # aka Type II Error



# evaluate model based on fit type
def evaluate(fit_type, y_true, y_pred, y_prob):

    results = {}

    if fit_type == "binaryclass":
        results["Accuracy"] = accuracy_score(y_true, y_pred)
        results["Balanced Accuracy"] = balanced_accuracy_score(y_true, y_pred)
        results["Kappa Score"] = cohen_kappa_score(y_true, y_pred)
        results["MCC"] = matthews_corrcoef(y_true, y_pred)
        results["ROC AUC Score"] = roc_auc_score(y_true, y_prob)
        results["Macro F1 Score"] = f1_score(y_true, y_pred, average="macro")
        results["Weighted F1 Score"] = f1_score(y_true, y_pred, average="weighted")
        results["Micro F1 Score"] = f1_score(y_true, y_pred, average="micro")
        results["Log Loss"] = log_loss(y_true, y_prob)
        results["Precision"] = precision_score(y_true, y_pred)
        results["Recall"] = recall_score(y_true, y_pred)   # aka Sensitivity
        results.update(classification_extra(y_true, y_pred))

        # visualizations
        sns_plot = sns.heatmap(confusion_matrix(y_true, y_pred), annot=True,
                    cmap=sns.light_palette((210, 90, 60), input="husl"))
        sns_plot.set_title('Confusion Matrix')
        sns_plot.set(xlabel="Predicted", ylabel="Actual")
        sns_plot.figure.savefig("logs/img/confusion-matrix.png")

    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Score'])

    logger.info(tabulate(results_df, headers="keys", tablefmt="psql"))





