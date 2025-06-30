from sklearn.ensemble import GradientBoostingClassifier
from utils.timer import timed


@timed
def run_gboost(x_train, y_train):
    print("→ [Gradient Boosting] Fitting model...")
    gbc = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=1.0,
        max_depth=1,
        random_state=0
    )
    gbc.fit(x_train, y_train)

    print("→ [Gradient Boosting] Predicting...")
    y_pred = gbc.predict(x_train)

    print("→ [Gradient Boosting] Done.")
    return y_pred, gbc
