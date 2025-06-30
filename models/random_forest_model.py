from sklearn.ensemble import RandomForestClassifier
from utils.timer import timed


@timed
def run_rforest(X_train, y_train):

    print("→ [Random Forest] Fitting model...")
    rf = RandomForestClassifier(max_depth=2, n_jobs=-1)
    rf.fit(X_train, y_train)

    print("→ [Random Forest] Predicting...")
    y_pred = rf.predict(X_train)

    print("→ [Random Forest] Done.")
    return y_pred, rf
