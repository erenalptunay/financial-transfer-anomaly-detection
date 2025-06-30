from pyod.models.lof import LOF
from utils.timer import timed


@timed
def run_lof(x, contamination=0.00172, n_neighbors=20):
    print("→ [LOF] Fitting and predicting...")

    model = LOF(
        contamination=contamination,
        n_neighbors=n_neighbors,
        n_jobs=-1
    )

    y_pred = model.fit_predict(x)
    scores = model.decision_scores_

    print("→ [LOF] Done.")
    return y_pred, scores, model
