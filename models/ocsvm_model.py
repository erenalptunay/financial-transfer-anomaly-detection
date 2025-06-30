from pyod.models.ocsvm import OCSVM
from utils.timer import timed
import pandas as pd

@timed
def run_ocsvm(x, contamination=0.00172, kernel="linear"):

    x = pd.DataFrame(x).sample(20000, random_state=42)

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
