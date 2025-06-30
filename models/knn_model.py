from pyod.models.knn import KNN
from utils.timer import timed


@timed
def run_knn(x, contamination=0.00172):
    print("→ [KNN] Fitting model...")
    model = KNN(
        contamination=contamination,
        n_neighbors=5,
        method='largest',
        n_jobs=-1
    )
    model.fit(x)

    print("→ [KNN] Predicting...")
    y_pred = model.predict(x)
    scores = model.decision_scores_

    print("→ [KNN] Done.")
    return y_pred, scores, model
