from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tabulate import tabulate
# from rich.progress import progress
import numpy as np

def evaluate_model_success(y_true, y_pred):

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    total = len(y_true)
    correct = np.sum(y_true == y_pred)

    # Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC AUC metric
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = None

    print(
        f"🎯 F1 Score              : %{f1 * 100:.2f}\n"
        f"🔍 Precision             : %{precision * 100:.2f}\n"
        f"📢 Recall                : %{recall * 100:.2f}"
    )
    print(f"📈 ROC-AUC               : %{auc * 100:.2f}" if auc is not None else "")
    print(
        f"✅ Accuracy              : %{accuracy * 100:.2f}\n"
        f"✔️ Correct predictions   : {correct} / {total}\n"
        f"❌ Incorrect predictions : {total - correct}"
    )

    # Confussion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    table = [
        ["Actual 0", f"{cm[0][0]}", f"{cm[0][1]}"],
        ["Actual 1", f"{cm[1][0]}", f"{cm[1][1]}"]
    ]
    headers = ["", "Predicted 0", "Predicted 1"]

    print("📊 Confusion Matrix:")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
