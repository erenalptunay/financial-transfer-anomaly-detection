from pyod.models.knn import KNN


def run_knn(X, contamination=0.00172):
    print("→ [KNN] Fitting model...")
    model = KNN(
        contamination=contamination,
        n_neighbors=5,
        method='mean',
        n_jobs=-1
    )
    model.fit(X)

    print("→ [KNN] Predicting...")
    y_pred = model.predict(X)
    scores = model.decision_scores_

    print("→ [KNN] Done.")
    return y_pred, scores, model
