from pyod.models.lof import LOF


def run_lof(X, contamination=0.00172, n_neighbors=20):
    print("→ [LOF] Fitting and predicting...")

    model = LOF(
        contamination=contamination,
        n_neighbors=n_neighbors,
        n_jobs=-1
    )

    y_pred = model.fit_predict(X)
    scores = model.decision_scores_

    print("→ [LOF] Done.")
    return y_pred, scores, model








    # lof = LOF(contamination=0.1)
    # lof.fit(X_train)
    # y_pred_train = lof.labels_
    # y_pred_test = lof.predict(X_test)
    # return y_pred_train, y_pred_test