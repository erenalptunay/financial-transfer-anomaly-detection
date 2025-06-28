from sklearn.metrics import accuracy_score
import numpy as np

def evaluate_model_success(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)
    total = len(y_true)
    correct = np.sum(y_true == y_pred)

    print(
        f"✔️ Correct predictions  : {correct} / {total}\n"
        f"❌ Mistaken predictions : {total - correct}\n"
        f"✅ Accuracy             : %{accuracy * 100:.2f}"
    )
