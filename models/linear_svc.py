from sklearn.svm import LinearSVC
from utils.timer import timed


@timed
def run_linear_svc(X_train, y_train):
    print("→ [Linear SVC] Fitting model...")

    model = LinearSVC(
        class_weight='balanced',
    )

    model.fit(X_train, y_train)

    print("→ [Linear SVC] Predicting...")
    y_pred = model.predict(X_train)

    print("→ [Linear SVC] Done.")
    return y_pred, model
