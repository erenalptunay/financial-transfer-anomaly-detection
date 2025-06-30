from pyod.models.iforest import IForest
from utils.timer import timed


@timed
def run_iforest(x, contamination=0.01, random_state=42):
    print("→ [IForest] Fitting model...")
    clf = IForest(contamination=contamination, random_state=random_state)
    clf.fit(x)

    print("→ [IForest] Predicting...")
    y_pred = clf.predict(x)
    scores = clf.decision_scores_

    print("→ [IForest] Done.")
    return y_pred, scores, clf
