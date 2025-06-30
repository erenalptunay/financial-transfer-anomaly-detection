from sklearn.svm import LinearSVC
from utils.timer import timed


@timed
def run_linear_svc(x_train, y_train):
    print("→ [Linear SVC] Fitting model...")

    model = LinearSVC(
        class_weight='balanced',
        C=0.1
    )

    model.fit(x_train, y_train)

    print("→ [Linear SVC] Predicting...")
    y_pred = model.predict(x_train)

    print("→ [Linear SVC] Done.")
    return y_pred, model
