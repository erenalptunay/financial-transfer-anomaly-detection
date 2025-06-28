from pyod.models.ocsvm import OCSVM
from utils.timer import timed


@timed
def run_ocsvm(X, contamination=0.00172, kernel="linear"):
    print("→ [OCSVM] Fitting model...")

    model = OCSVM(
        contamination=contamination,
        kernel=kernel,
        nu=0.5,
        gamma='auto'
    )

    model.fit(X)

    print("→ [OCSVM] Predicting...")
    y_pred = model.predict(X)
    scores = model.decision_scores_

    print("→ [OCSVM] Done.")
    return y_pred, scores, model