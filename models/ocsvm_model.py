from pyod.models.ocsvm import OCSVM
from utils.timer import timed


@timed
def run_ocsvm(x, contamination=0.00172, kernel="linear"):
    print("→ [OCSVM] Fitting model...")

    model = OCSVM(
        contamination=contamination,
        kernel=kernel,
        nu=0.5,
        gamma='auto'
    )

    model.fit(x)

    print("→ [OCSVM] Predicting...")
    y_pred = model.predict(x)
    scores = model.decision_scores_

    print("→ [OCSVM] Done.")
    return y_pred, scores, model