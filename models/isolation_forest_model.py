from pyod.models.iforest import IForest


def run_iforest(X, contamination=0.002, random_state=42):
    print("→ [IForest] Fitting model...")
    clf = IForest(contamination=contamination, random_state=random_state)
    clf.fit(X)

    print("→ [IForest] Predicting...")
    y_pred = clf.predict(X)
    scores = clf.decision_scores_

    print("→ [IForest] Done.")
    return y_pred, scores, clf
