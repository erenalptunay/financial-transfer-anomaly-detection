from pyod.models.iforest import IForest


def run_iforest(X, contamination=0.00172, random_state=42):

    clf = IForest(contamination=contamination, random_state=random_state)
    clf.fit(X)

    y_pred = clf.predict(X)
    scores = clf.decision_scores_

    return y_pred, scores, clf
