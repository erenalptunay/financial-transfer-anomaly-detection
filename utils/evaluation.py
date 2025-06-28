from sklearn.metrics import accuracy_score


def evaluate_model_success(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)
    total = len(y_true)
    correct = (y_true == y_pred).sum()

    print(f"""
    ✔️ Correct predictions: {correct} / {total}
    ❌ Mistaken predictions: {total - correct}
    ✅ Accuracy: %{accuracy * 100:.2f}
    """)
