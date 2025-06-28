from pyod.models.knn import KNN

def run_knn(X, contamination=0.00172):
    model = KNN(contamination = contamination)
    model.fit(X)

    y_pred = model.predict(X)
    scores = model.decision_scores_

    return y_pred, scores, model