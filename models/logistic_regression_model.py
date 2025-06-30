from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from utils.timer import timed


@timed
def run_logreg(X_train, y_train, X_test):
    print("→ [LogReg] Scaling data using RobustScaler...")

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("→ [LogReg] Fitting Logistic Regression model...")

    model = LogisticRegression(n_jobs = -1)
    model.fit(X_train_scaled, y_train)

    print("→ [LogReg] Predicting on test set...")
    y_pred = model.predict(X_test_scaled)

    print("→ [LogReg] Done.")
    return y_pred, model, scaler